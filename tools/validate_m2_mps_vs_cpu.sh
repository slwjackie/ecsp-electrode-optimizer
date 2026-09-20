#!/usr/bin/env bash
set -euo pipefail
echo "ERROR: CPU-vs-MPS parity is not meaningful after the v7.9.0 surface-contact model correction." >&2
echo "The legacy MPS engine uses different electrode-mask physics. Validate the C++ engine with tools/validate_surface_contact_cpp.sh." >&2
exit 2
