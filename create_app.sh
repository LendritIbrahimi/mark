#!/bin/bash
# Build a macOS .app bundle so mark launches without a Terminal window.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$DIR/mark.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/MacOS/mark" << LAUNCHER
#!/bin/bash
cd "$DIR"
exec "$DIR/.venv/bin/python" -m ui.app 2>>"$DIR/mark_app.log"
LAUNCHER
chmod +x "$APP/Contents/MacOS/mark"

cat > "$APP/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>mark</string>
  <key>CFBundleIdentifier</key>
  <string>dev.mark.agent</string>
  <key>CFBundleExecutable</key>
  <string>mark</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>NSScreenCaptureUsageDescription</key>
  <string>mark needs screen access to see what's on your desktop</string>
  <key>NSAppleEventsUsageDescription</key>
  <string>mark needs automation access to control apps on your behalf</string>
</dict>
</plist>
PLIST

xattr -cr "$APP"
codesign --deep --force --sign - "$APP"
echo "Created $APP"
