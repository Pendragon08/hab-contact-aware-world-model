"""Verify a draft by release ID; GitHub's tag endpoint may not expose drafts."""
import argparse
import json
import subprocess

from verify_github_release import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--bundles", nargs="+", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()
    release = json.loads(subprocess.check_output(
        ["gh", "api", f"repos/Pendragon08/hab-contact-aware-world-model/releases/{args.id}"],
        text=True, encoding="utf-8",
    ))
    assert release["id"] == args.id and release["tag_name"] == args.tag and release["draft"]
    expected = []
    for name in args.bundles:
        inventory = ROOT / "release" / f"{name}.release.json"
        data = json.loads(inventory.read_text(encoding="utf-8"))
        expected += [inventory, ROOT / "release" / data["manifest"]]
        expected += [ROOT / "release" / part["name"] for part in data["parts"]]
    expected += [ROOT / name for name in args.extra]
    assets = {a["name"]: a for a in release["assets"]}
    assert set(assets) == {p.name for p in expected}
    rows = []
    for path in expected:
        asset = assets[path.name]
        digest = sha(path)
        assert asset["state"] == "uploaded" and asset["size"] == path.stat().st_size, path.name
        assert asset["digest"] == "sha256:" + digest, path.name
        rows.append({"name": path.name, "bytes": asset["size"], "sha256": digest})
    result = {"passed": True, "release_id": args.id, "tag": args.tag, "is_draft": True, "assets_verified": rows}
    (ROOT / "reports" / f"github_{args.tag}_draft_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DRAFT_ASSETS_VERIFIED", len(rows))


if __name__ == "__main__":
    main()
