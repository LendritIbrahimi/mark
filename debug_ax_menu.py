"""Diagnostic: deep-probe where Safari hides context menu AX elements.

Right-click in Safari to open a context menu, then run:
    .venv/bin/python debug_ax_menu.py 5
"""

import sys
import time

from AppKit import NSApplication, NSApplicationActivationPolicyProhibited
NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyProhibited)

from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyAttributeNames,
    kAXErrorSuccess,
)
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)


def _ax(el, attr):
    err, val = AXUIElementCopyAttributeValue(el, attr, None)
    return val if err == kAXErrorSuccess else None


def _attrs(el):
    err, names = AXUIElementCopyAttributeNames(el, None)
    return list(names) if err == kAXErrorSuccess and names else []


def _role(el):
    return _ax(el, "AXRole") or "?"


def _label(el):
    for a in ("AXTitle", "AXDescription", "AXValue", "AXRoleDescription"):
        v = _ax(el, a)
        if v:
            return str(v)[:80]
    return ""


def _walk(el, depth=0, max_depth=3, counter=None, look_for_menu=True):
    if counter is None:
        counter = [0]
    if depth > max_depth or counter[0] > 500:
        return
    counter[0] += 1
    role = _role(el)
    lbl = _label(el)
    indent = "  " * depth
    tag = f' "{lbl}"' if lbl else ""
    marker = " <<<< MENU FOUND!" if role in ("AXMenu", "AXMenuItem") else ""
    print(f"{indent}{role}{tag}{marker}")
    for c in (_ax(el, "AXChildren") or []):
        _walk(c, depth + 1, max_depth, counter, look_for_menu)


def main():
    delay = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Right-click now! Reading in {delay}s...")
    for i in range(delay, 0, -1):
        print(f"  {i}s ...")
        time.sleep(1)

    # --- 1. Confirm menu window is on screen ---
    print("\n=== CGWindowList: menu-level windows (layer >= 100) ===")
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    menu_window = None
    for w in windows:
        layer = w.get("kCGWindowLayer", 0)
        if layer >= 100:
            pid = w.get("kCGWindowOwnerPID", 0)
            name = w.get("kCGWindowOwnerName", "?")
            bounds = w.get("kCGWindowBounds", {})
            wid = w.get("kCGWindowNumber", 0)
            print(f"  PID={pid} WID={wid} layer={layer} owner={name} bounds={bounds}")
            if 100 <= layer <= 200 and name == "Safari":
                menu_window = w

    if not menu_window:
        print("\n** No Safari menu window detected. Was the context menu open? **")
        return

    safari_pid = menu_window["kCGWindowOwnerPID"]
    print(f"\nSafari menu window confirmed (PID={safari_pid})")

    # --- 2. Enumerate ALL attributes on Safari app element ---
    app = AXUIElementCreateApplication(safari_pid)
    print(f"\n=== ALL attributes on AXApplication (Safari) ===")
    for attr in sorted(_attrs(app)):
        val = _ax(app, attr)
        if isinstance(val, (str, int, float, bool)):
            print(f"  {attr} = {val}")
        elif isinstance(val, (list, tuple)):
            print(f"  {attr} = [{len(val)} items]")
            for i, item in enumerate(val[:10]):
                r = _role(item) if hasattr(item, '__class__') and 'AXUIElement' in str(type(item)) else str(item)[:80]
                l = _label(item) if hasattr(item, '__class__') and 'AXUIElement' in str(type(item)) else ""
                tag = f' "{l}"' if l else ""
                print(f"    [{i}] {r}{tag}")
        elif val is not None:
            tp = type(val).__name__
            print(f"  {attr} = <{tp}>")

    # --- 3. Check AXWindows vs AXChildren ---
    print(f"\n=== AXWindows attribute ===")
    ax_windows = _ax(app, "AXWindows") or []
    for i, w in enumerate(ax_windows):
        print(f"  [{i}] {_role(w)} \"{_label(w)}\"")
        for attr in _attrs(w):
            v = _ax(w, attr)
            if isinstance(v, (str, int, float, bool)):
                if attr in ("AXRole", "AXTitle", "AXSubrole", "AXRoleDescription"):
                    continue
                print(f"        {attr} = {v}")

    # --- 4. Walk AXMenuBar deeply ---
    print(f"\n=== AXMenuBar walk (depth=3) ===")
    menubar = _ax(app, "AXMenuBar")
    if menubar:
        mb_children = _ax(menubar, "AXChildren") or []
        print(f"  MenuBar has {len(mb_children)} children")
        for i, c in enumerate(mb_children[:5]):
            print(f"  [{i}] {_role(c)} \"{_label(c)}\"")
    else:
        print("  (no AXMenuBar)")

    # --- 5. Walk the main window's children looking for AXMenu ---
    print(f"\n=== Walking AXWindow children (depth=4) looking for AXMenu ===")
    ax_children = _ax(app, "AXChildren") or []
    for c in ax_children:
        role = _role(c)
        if role == "AXWindow":
            win_children = _ax(c, "AXChildren") or []
            print(f"  Window \"{_label(c)}\" has {len(win_children)} children:")
            for i, wc in enumerate(win_children):
                r = _role(wc)
                l = _label(wc)
                marker = " <<<< MENU!" if r in ("AXMenu", "AXMenuItem") else ""
                print(f"    [{i}] {r} \"{l}\"{marker}")

    # --- 6. Check AXFocusedUIElement and its attributes ---
    print(f"\n=== Focused element deep-probe ===")
    focused = _ax(app, "AXFocusedUIElement")
    if focused:
        print(f"  Role: {_role(focused)}")
        print(f"  Label: {_label(focused)}")
        print(f"  Attributes: {_attrs(focused)}")

    # --- 7. System-wide: check kAXFocusedUIElementAttribute ---
    sw = AXUIElementCreateSystemWide()
    sys_focused = _ax(sw, "AXFocusedUIElement")
    print(f"\n=== System-wide focused ===")
    if sys_focused:
        print(f"  {_role(sys_focused)} \"{_label(sys_focused)}\"")

    # --- 8. Try creating AXUIElement for every on-screen PID ---
    print(f"\n=== Check ALL on-screen app PIDs for AXMenu children ===")
    all_pids = set()
    for w in windows:
        all_pids.add(w.get("kCGWindowOwnerPID", 0))
    for pid in sorted(all_pids):
        if pid == 0:
            continue
        a = AXUIElementCreateApplication(pid)
        children = _ax(a, "AXChildren") or []
        menus = [c for c in children if _role(c) in ("AXMenu", "AXMenuItem")]
        if menus:
            name = _ax(a, "AXTitle") or f"PID={pid}"
            print(f"  {name}: found {len(menus)} AXMenu children!")
            for m in menus:
                _walk(m, depth=2, max_depth=4)


if __name__ == "__main__":
    main()
