#!/bin/bash
set -e

# Tell GDM/Mutter to use the iGPU (card1) as primary display GPU
# instead of the MI50 (card0)

sudo mkdir -p /etc/X11/xorg.conf.d
sudo tee /etc/udev/rules.d/61-mutter-primary-gpu.rules << 'EOF'
ENV{DEVNAME}=="/dev/dri/card1", TAG+="mutter-device-preferred-primary"
EOF

echo "Done. Reboot for changes to take effect."
