#!/usr/bin/env bash
#
# Installs the system shared libraries this module links against at runtime but
# does not ship inside its own bundle.
#
# Viam runs this once per module version at install time, before the module is
# started for the first time (the "first_run" key in meta.json). Whatever this
# script prints to stdout is logged at INFO level in the machine's logs, and
# stderr is logged at WARN, so the messages below are visible with
# `viam machines part logs` or in the app.
#
# EXIT CODE: this script exits non-zero when the libraries are missing and could
# not be installed. That deliberately aborts the machine's reconfiguration -- the
# RDK keeps the previous working config, leaves already-running modules running,
# and marks this module's package as failed -- rather than letting the machine
# come up with a module that cannot start. Because no success marker is written
# on failure, first_run is retried automatically once the host is fixed.
#
# THROTTLING: an aborted reconfiguration is retried on every cloud config
# refresh (~10s by default), because the RDK only advances its stored config
# after the first_run phase succeeds. Re-running a package install that often
# would hold the package-manager lock and fight the very operator trying to fix
# the host, so the install attempt is rate-limited to once per
# RETRY_INTERVAL_SECONDS. The diagnostic below is still printed, and the exit
# code is still non-zero, on every single run -- only the install is skipped.

set -uo pipefail

readonly MODULE_NAME="viam-labs:YOLOv8"

# Sonames the bundled Python extensions link but do not vendor. Taken from the
# DT_NEEDED entries of the ELF objects inside the opencv-python wheel -- the
# same set on x86_64 and aarch64. They all arrive via the Qt GUI stack that the
# non-headless opencv-python build depends on, which is why a module that never
# opens a window still needs libGL.
readonly REQUIRED_SONAMES=(
    libGL.so.1
    libICE.so.6
    libSM.so.6
    libX11.so.6
    libXext.so.6
    libglib-2.0.so.0
    libgthread-2.0.so.0
    libxcb.so.1
    libz.so.1
)

log() { echo "[first_run] $*"; }
warn() { echo "[first_run] $*" >&2; }

# Command prefix used for privileged operations; set once in main().
SUDO=""
# Space-separated list of packages this script tried to install, quoted back to
# the operator if it could not finish the job.
MANUAL_PACKAGES=""
# Filled in by resolve_packages().
PACKAGES=()
UNRESOLVED=()

# Minimum gap between real install attempts. See THROTTLING above.
readonly RETRY_INTERVAL_SECONDS=600

# Where the last failed attempt is recorded. Placed as a sibling of the module
# directory, mirroring where the RDK keeps its own `.first_run_succeeded`
# marker, because that directory is root-owned. A world-writable location such
# as /tmp would let any local user pre-create this path as a symlink and have
# us truncate the target while running as root. When VIAM_MODULE_ROOT is not set
# -- someone running this script by hand, outside the RDK -- throttling is
# simply disabled, which is the right behavior for a manual invocation.
ATTEMPT_STAMP=""
if [ -n "${VIAM_MODULE_ROOT:-}" ]; then
    ATTEMPT_STAMP="${VIAM_MODULE_ROOT%/}.first_run_last_failure"
fi

now_seconds() { date +%s; }

record_failed_attempt() {
    [ -n "$ATTEMPT_STAMP" ] || return 0
    now_seconds >"$ATTEMPT_STAMP" 2>/dev/null || true
}

clear_failed_attempt() {
    [ -n "$ATTEMPT_STAMP" ] || return 0
    rm -f "$ATTEMPT_STAMP" 2>/dev/null || true
}

# True when a previous attempt failed recently enough that retrying the package
# manager now would just add lock contention.
attempt_throttled() {
    local last age
    [ -n "$ATTEMPT_STAMP" ] || return 1
    [ -f "$ATTEMPT_STAMP" ] || return 1
    last=$(cat "$ATTEMPT_STAMP" 2>/dev/null) || return 1
    case "$last" in '' | *[!0-9]*) return 1 ;; esac
    age=$(($(now_seconds) - last))
    [ "$age" -ge 0 ] && [ "$age" -lt "$RETRY_INTERVAL_SECONDS" ]
}

# --------------------------------------------------------------------------- #
# Detecting what is already present
# --------------------------------------------------------------------------- #

# ldconfig's cache is architecture-aware, so a soname lookup resolves correctly
# on amd64, arm64 and anything else without this script having to know about
# /usr/lib/<triplet> layouts.
soname_present() {
    local soname=$1 dir
    if command -v ldconfig >/dev/null 2>&1; then
        if ldconfig -p 2>/dev/null | awk -v s="$soname" '$1 == s { hit = 1 } END { exit !hit }'; then
            return 0
        fi
    fi
    # Fallback for hosts with no ldconfig (or an empty cache).
    for dir in /lib /usr/lib /usr/local/lib \
        "/lib/$(uname -m)-linux-gnu" "/usr/lib/$(uname -m)-linux-gnu"; do
        [ -e "$dir/$soname" ] && return 0
    done
    return 1
}

missing_sonames() {
    local soname
    for soname in "${REQUIRED_SONAMES[@]}"; do
        soname_present "$soname" || echo "$soname"
    done
}

# --------------------------------------------------------------------------- #
# soname -> package name, per package manager
#
# Where a distro has renamed a package, the candidates are listed newest-first
# and the first one the package manager actually knows about is used. That is
# what lets this work across Debian/Ubuntu's 64-bit time_t rename
# (libasound2t64 vs libasound2, libglib2.0-0t64 vs libglib2.0-0) without
# pinning a distro release.
# --------------------------------------------------------------------------- #
packages_for_soname() {
    local mgr=$1 soname=$2
    case "$mgr" in
    apt)
        case "$soname" in
        libGL.so.1) echo "libgl1 libgl1-mesa-glx" ;;
        libICE.so.6) echo "libice6" ;;
        libSM.so.6) echo "libsm6" ;;
        libX11.so.6) echo "libx11-6" ;;
        libXext.so.6) echo "libxext6" ;;
        libglib-2.0.so.0 | libgthread-2.0.so.0) echo "libglib2.0-0t64 libglib2.0-0" ;;
        libxcb.so.1) echo "libxcb1" ;;
        libz.so.1) echo "zlib1g" ;;
        libasound.so.2) echo "libasound2t64 libasound2" ;;
        esac
        ;;
    dnf | yum)
        case "$soname" in
        libGL.so.1) echo "mesa-libGL" ;;
        libICE.so.6) echo "libICE" ;;
        libSM.so.6) echo "libSM" ;;
        libX11.so.6) echo "libX11" ;;
        libXext.so.6) echo "libXext" ;;
        libglib-2.0.so.0 | libgthread-2.0.so.0) echo "glib2" ;;
        libxcb.so.1) echo "libxcb" ;;
        libz.so.1) echo "zlib-ng-compat zlib" ;;
        libasound.so.2) echo "alsa-lib" ;;
        esac
        ;;
    zypper)
        case "$soname" in
        libGL.so.1) echo "Mesa-libGL1" ;;
        libICE.so.6) echo "libICE6" ;;
        libSM.so.6) echo "libSM6" ;;
        libX11.so.6) echo "libX11-6" ;;
        libXext.so.6) echo "libXext6" ;;
        libglib-2.0.so.0 | libgthread-2.0.so.0) echo "glib2" ;;
        libxcb.so.1) echo "libxcb1" ;;
        libz.so.1) echo "libz1" ;;
        libasound.so.2) echo "libasound2" ;;
        esac
        ;;
    pacman)
        case "$soname" in
        libGL.so.1) echo "libglvnd" ;;
        libICE.so.6) echo "libice" ;;
        libSM.so.6) echo "libsm" ;;
        libX11.so.6) echo "libx11" ;;
        libXext.so.6) echo "libxext" ;;
        libglib-2.0.so.0 | libgthread-2.0.so.0) echo "glib2" ;;
        libxcb.so.1) echo "libxcb" ;;
        libz.so.1) echo "zlib" ;;
        libasound.so.2) echo "alsa-lib" ;;
        esac
        ;;
    apk)
        case "$soname" in
        libGL.so.1) echo "mesa-gl" ;;
        libICE.so.6) echo "libice" ;;
        libSM.so.6) echo "libsm" ;;
        libX11.so.6) echo "libx11" ;;
        libXext.so.6) echo "libxext" ;;
        libglib-2.0.so.0 | libgthread-2.0.so.0) echo "glib" ;;
        libxcb.so.1) echo "libxcb" ;;
        libz.so.1) echo "zlib" ;;
        libasound.so.2) echo "alsa-lib" ;;
        esac
        ;;
    esac
}

detect_package_manager() {
    local mgr
    for mgr in apt-get dnf yum zypper pacman apk; do
        if command -v "$mgr" >/dev/null 2>&1; then
            # Normalize apt-get to the "apt" key used by the tables above.
            [ "$mgr" = "apt-get" ] && mgr=apt
            echo "$mgr"
            return 0
        fi
    done
    return 1
}

# Only probed for the managers where a cheap, reliable query exists. Anywhere
# else we optimistically assume the mapped name is right and let the install
# surface the error.
package_available() {
    local mgr=$1 pkg=$2
    case "$mgr" in
    apt) apt-cache show "$pkg" >/dev/null 2>&1 ;;
    dnf) dnf --quiet info "$pkg" >/dev/null 2>&1 ;;
    yum) yum --quiet info "$pkg" >/dev/null 2>&1 ;;
    pacman) pacman -Si "$pkg" >/dev/null 2>&1 ;;
    *) return 0 ;;
    esac
}

# Failure is tolerated everywhere it is called: a broken or unreachable
# third-party source (an expired signing key elsewhere in sources.list.d, say)
# must not stop us installing libraries that come from the distro's own
# archives.
refresh_package_lists() {
    local mgr=$1
    case "$mgr" in
    apt) $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update ;;
    dnf) $SUDO dnf --quiet makecache ;;
    yum) $SUDO yum --quiet makecache ;;
    zypper) $SUDO zypper --non-interactive refresh ;;
    pacman) $SUDO pacman -Sy --noconfirm ;;
    apk) $SUDO apk update ;;
    esac || warn "refreshing package lists reported errors; continuing anyway"
}

install_packages() {
    local mgr=$1
    shift
    case "$mgr" in
    apt) $SUDO env DEBIAN_FRONTEND=noninteractive \
        apt-get install -y --no-install-recommends "$@" ;;
    dnf) $SUDO dnf install -y "$@" ;;
    yum) $SUDO yum install -y "$@" ;;
    zypper) $SUDO zypper --non-interactive install "$@" ;;
    pacman) $SUDO pacman -Sy --noconfirm --needed "$@" ;;
    apk) $SUDO apk add --no-cache "$@" ;;
    esac
}

host_description() {
    local pretty=""
    if [ -r /etc/os-release ]; then
        # shellcheck disable=SC1091
        pretty=$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-}")
    fi
    echo "${pretty:-$(uname -s)} ($(uname -m))"
}

# The preferred (first) candidate for each soname, ignoring what the package
# cache says is available. Used only to build the suggested manual command when
# nothing could be resolved for real.
preferred_packages_for() {
    local mgr=$1
    shift
    local soname candidate list=""
    for soname in "$@"; do
        candidate=$(packages_for_soname "$mgr" "$soname")
        candidate=${candidate%% *}
        [ -z "$candidate" ] && continue
        case " $list " in
        *" $candidate "*) ;;
        *) list="$list $candidate" ;;
        esac
    done
    echo "${list# }"
}

# Printed whenever we give up, so an operator has one copy-pasteable command.
report_unresolved() {
    local mgr=$1 reason=$2
    shift 2
    local still_missing=("$@")
    local packages="$MANUAL_PACKAGES" guessed=0

    # An unprivileged run never gets as far as resolving package names, so fall
    # back to the preferred candidate per soname. Better an approximate command
    # than none at all.
    if [ -z "$packages" ] && [ -n "$mgr" ]; then
        packages=$(preferred_packages_for "$mgr" "${still_missing[@]}")
        guessed=1
    fi

    warn "-----------------------------------------------------------------"
    warn "$MODULE_NAME: missing system libraries could not be installed"
    warn ""
    warn "Host:      $(host_description)"
    warn "Reason:    $reason"
    warn "Missing:   ${still_missing[*]}"
    if [ -n "$mgr" ] && [ -n "$packages" ]; then
        warn ""
        if [ "$guessed" -eq 1 ]; then
            warn "Install these by hand (package names are this distro's usual"
            warn "ones; they were not verified against the package cache), then"
            warn "restart viam-server on this machine:"
        else
            warn "Install them by hand, then restart viam-server on this machine:"
        fi
        warn ""
        case "$mgr" in
        apt) warn "    sudo apt-get update && sudo apt-get install -y $packages" ;;
        dnf) warn "    sudo dnf install -y $packages" ;;
        yum) warn "    sudo yum install -y $packages" ;;
        zypper) warn "    sudo zypper install $packages" ;;
        pacman) warn "    sudo pacman -Sy --needed $packages" ;;
        apk) warn "    sudo apk add $packages" ;;
        esac
    fi
    warn ""
    warn "This module cannot start without these libraries, so first_run is"
    warn "failing on purpose. The machine will keep running its previous"
    warn "configuration and will not apply the new one until this is fixed."
    warn "Once the libraries are installed the machine retries automatically."
    warn "-----------------------------------------------------------------"
}

# Maps each missing soname to a package this host actually offers, leaving the
# result in the PACKAGES and UNRESOLVED globals.
resolve_packages() {
    local mgr=$1
    shift
    local soname candidates candidate chosen
    PACKAGES=()
    UNRESOLVED=()
    for soname in "$@"; do
        candidates=$(packages_for_soname "$mgr" "$soname")
        if [ -z "$candidates" ]; then
            UNRESOLVED+=("$soname")
            continue
        fi
        chosen=""
        for candidate in $candidates; do
            if package_available "$mgr" "$candidate"; then
                chosen=$candidate
                break
            fi
        done
        if [ -z "$chosen" ]; then
            UNRESOLVED+=("$soname")
            continue
        fi
        # De-duplicate: one package can provide several sonames, e.g.
        # libglib2.0-0t64 ships both libglib-2.0.so.0 and libgthread-2.0.so.0.
        case " ${PACKAGES[*]-} " in
        *" $chosen "*) ;;
        *) PACKAGES+=("$chosen") ;;
        esac
    done
}

main() {
    if [ "$(uname -s)" != "Linux" ]; then
        log "host is $(uname -s), not Linux: nothing to install for $MODULE_NAME"
        exit 0
    fi

    local missing=()
    while IFS= read -r line; do
        [ -n "$line" ] && missing+=("$line")
    done < <(missing_sonames)

    if [ ${#missing[@]} -eq 0 ]; then
        log "$MODULE_NAME: all required system libraries are already present"
        clear_failed_attempt
        exit 0
    fi
    log "$MODULE_NAME: missing system libraries: ${missing[*]}"

    local mgr=""
    if ! mgr=$(detect_package_manager); then
        report_unresolved "" "no supported package manager found (looked for apt-get, dnf, yum, zypper, pacman, apk)" "${missing[@]}"
        record_failed_attempt
        exit 1
    fi

    # Still fails loudly below -- only the package-manager work is skipped. See
    # THROTTLING at the top of this file.
    if attempt_throttled; then
        report_unresolved "$mgr" "a previous install attempt failed less than ${RETRY_INTERVAL_SECONDS}s ago; not retrying the package manager yet" "${missing[@]}"
        exit 1
    fi
    log "using package manager: $mgr"

    # Escalation is settled before anything that needs it, because refreshing
    # package lists below is itself a privileged operation.
    if [ "$(id -u)" -eq 0 ]; then
        SUDO=""
    elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        SUDO="sudo -n"
        log "not running as root; escalating with passwordless sudo"
    else
        report_unresolved "$mgr" "running as uid $(id -u) and passwordless sudo is unavailable" "${missing[@]}"
        record_failed_attempt
        exit 1
    fi

    resolve_packages "$mgr" "${missing[@]}"
    # An empty package cache makes every availability probe come up empty, which
    # is the normal state of a freshly provisioned host or a container built
    # with `apt clean`. Refresh once and re-resolve before concluding that a
    # library is genuinely unavailable.
    if [ ${#UNRESOLVED[@]} -gt 0 ]; then
        log "no package found yet for: ${UNRESOLVED[*]} -- refreshing package lists and retrying"
        refresh_package_lists "$mgr"
        resolve_packages "$mgr" "${missing[@]}"
    fi
    if [ ${#UNRESOLVED[@]} -gt 0 ]; then
        warn "no package on this distro provides: ${UNRESOLVED[*]}"
    fi

    if [ ${#PACKAGES[@]} -eq 0 ]; then
        report_unresolved "$mgr" "could not map the missing libraries to any available package on this distro" "${missing[@]}"
        record_failed_attempt
        exit 1
    fi

    # Read by report_unresolved so the manual command it prints is exactly what
    # this script tried to install.
    MANUAL_PACKAGES="${PACKAGES[*]}"
    log "installing: $MANUAL_PACKAGES"

    if ! install_packages "$mgr" "${PACKAGES[@]}"; then
        warn "$mgr failed to install: $MANUAL_PACKAGES"
    fi

    # Refresh the loader cache so the verification below sees what was installed.
    $SUDO ldconfig 2>/dev/null || true

    local still_missing=()
    while IFS= read -r line; do
        [ -n "$line" ] && still_missing+=("$line")
    done < <(missing_sonames)

    if [ ${#still_missing[@]} -eq 0 ]; then
        log "$MODULE_NAME: all required system libraries are now present"
        clear_failed_attempt
        exit 0
    fi

    report_unresolved "$mgr" "still missing after installing $MANUAL_PACKAGES" "${still_missing[@]}"
    record_failed_attempt
    exit 1
}

main "$@"
