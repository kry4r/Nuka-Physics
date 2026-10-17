"""Scene viewport captures for the view smoke, kept by name across its python steps."""
import pathlib

SHOTS = {}
OUTLINE = (255, 158, 51)


def _ppm(data):
    fields, i = [], 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    return int(fields[1]), int(fields[2]), data[i + 1:]


def shot(app, name, out):
    """Captures the middle of the next drawn frame's scene viewport, clear of its HUD and toasts, as
    SHOTS[name]: a list of RGB tuples, every third pixel of every third row."""
    path = pathlib.Path(out) / f"view_{name}.ppm"
    path.unlink(missing_ok=True)
    for _ in range(4):
        app.update(True)
    app.actions.run("view.screenshot", path=str(path))
    for _ in range(600):
        if path.exists():
            break
        app.update(True)
    w, _, rgb = _ppm(path.read_bytes())
    v = app.actions.run("view.stats")["viewport"]
    pixels = []
    for y in range(v["y"] + v["height"] // 5, v["y"] + v["height"] * 4 // 5, 3):
        row = (y * w + v["x"]) * 3
        for x in range(v["width"] * 3 // 20, v["width"] * 17 // 20, 3):
            k = row + x * 3
            pixels.append((rgb[k], rgb[k + 1], rgb[k + 2]))
    SHOTS[name] = pixels
    return pixels


def _close(p, q, tol):
    return all(abs(a - b) <= tol for a, b in zip(p, q))


def near(name, rgb, tol):
    """The fraction of SHOTS[name] within tol of rgb on every channel."""
    px = SHOTS[name]
    return sum(_close(p, rgb, tol) for p in px) / len(px)


def background(name):
    """The most frequent colour of a shot."""
    counts = {}
    for p in SHOTS[name]:
        counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.get)


def foreground(name, bg):
    """The indices of a shot's pixels away from the background colour bg."""
    return [i for i, p in enumerate(SHOTS[name]) if not _close(p, bg, 3)]


def changed(a, b, where):
    """The fraction of the pixels at `where` that two shots draw differently."""
    pa, pb = SHOTS[a], SHOTS[b]
    return sum(not _close(pa[i], pb[i], 6) for i in where) / len(where)


def unit_normals(name, where):
    """The fraction of the pixels at `where` whose colours decode as unit normals, 2 rgb / 255 - 1."""
    ok = 0
    for i in where:
        n = [2.0 * c / 255.0 - 1.0 for c in SHOTS[name][i]]
        ok += abs(sum(c * c for c in n) - 1.0) < 0.12
    return ok / len(where)
