#!/usr/bin/env bash
# Launch Isaac Sim's bundled python via the dynamic loader. If the shared install's
# executables carry the setgid bit, a direct exec runs in secure-execution mode and
# drops LD_PRELOAD/LD_LIBRARY_PATH ("libcarb.so: cannot open shared object file").
# Going through ld.so ignores the bit. Selected automatically by scripts/run_isaac.sh.
exec /lib64/ld-linux-x86-64.so.2 "${ISAAC_PATH:?}/kit/python/bin/python3" "$@"
