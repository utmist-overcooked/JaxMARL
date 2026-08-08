#!/usr/bin/env bash
# Reload the NVIDIA kernel modules so they match the installed userspace libs.
#
# Symptom this fixes:
#   nvidia-smi -> "Failed to initialize NVML: Driver/library version mismatch"
#   JAX/CUDA   -> "kernel version 535.309.1 does not match DSO version 580.173.2"
#
# Cause: the driver was upgraded 535 -> 580, which replaced the userspace
# libraries, but the running kernel still has the old 535 module loaded. Linux
# cannot swap a module that is in use, so it stays until reloaded or rebooted.
#
# This is a reboot-free fix. It DOES restart the graphical session: anyone
# logged in at the desktop loses it. SSH sessions are unaffected.
#
# Run:  sudo bash scripts/fix_nvidia_driver.sh
set -uo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Must run as root:  sudo bash $0" >&2
  exit 1
fi

echo "== before =="
echo -n "  kernel module: "
grep -oE "Kernel Module +[0-9.]+" /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "?"
echo -n "  userspace lib: "
readlink /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null || echo "?"

# Refuse to run if a compute process still holds the GPU - unloading under it
# would kill someone's job.
busy=""
for p in /proc/[0-9]*; do
  if ls -l "$p/fd" 2>/dev/null | grep -q "/dev/nvidia"; then
    busy="$busy $(basename "$p")($(cat "$p/comm" 2>/dev/null))"
  fi
done
if [ -n "$busy" ]; then
  echo "ABORT: processes still using the GPU:$busy" >&2
  echo "Stop them first, then re-run." >&2
  exit 1
fi

DM=""
for svc in gdm3 gdm lightdm sddm; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then DM="$svc"; break; fi
done

restore() {
  if [ -n "$DM" ]; then
    echo "-- restarting $DM"
    systemctl start "$DM" 2>/dev/null || true
  fi
}

if [ -n "$DM" ]; then
  echo "== stopping display manager ($DM) =="
  systemctl stop "$DM" || { echo "could not stop $DM" >&2; exit 1; }
  sleep 3
fi

echo "== unloading nvidia modules =="
ok=1
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  if lsmod | grep -q "^$m "; then
    modprobe -r "$m" 2>&1 || { echo "  failed to unload $m"; ok=0; }
  fi
done

if [ "$ok" -ne 1 ] || lsmod | grep -q "^nvidia "; then
  echo "ABORT: nvidia module still loaded - something is holding it." >&2
  lsmod | grep -E "^nvidia" >&2
  restore
  echo "Nothing changed; a reboot is the remaining option." >&2
  exit 1
fi

echo "== loading the new module =="
modprobe nvidia || { echo "modprobe nvidia FAILED" >&2; restore; exit 1; }
modprobe nvidia_uvm 2>/dev/null || true

restore

echo "== after =="
echo -n "  kernel module: "
grep -oE "Kernel Module +[0-9.]+" /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "?"
echo
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
  echo
  echo "SUCCESS - GPU is usable again."
else
  echo "nvidia-smi still failing:" >&2
  nvidia-smi 2>&1 | head -3 >&2
  echo "A reboot will resolve it." >&2
  exit 1
fi
