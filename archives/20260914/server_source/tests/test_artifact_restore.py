import contextlib,hashlib,io,json,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from archive_artifacts import build

class RestoreIntegrity(unittest.TestCase):
    def fixture(self,folder):
        root=folder/'source';(root/'artifacts').mkdir(parents=True)
        (root/'artifacts/记录.txt').write_text('实验记录\n',encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):build(root,'tiny',['artifacts'],500)
        return root
    def run_restore(self,source,target):
        return subprocess.run([sys.executable,str(ROOT/'scripts/restore_artifacts.py'),'--inventory',str(source/'release/tiny.release.json'),'--root',str(target)],capture_output=True,text=True)
    def test_roundtrip_and_idempotent_verification(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=self.fixture(p);dst=p/'restored'
            self.assertEqual(self.run_restore(src,dst).returncode,0)
            self.assertEqual((dst/'artifacts/记录.txt').read_text(encoding='utf-8'),'实验记录\n')
            self.assertEqual(self.run_restore(src,dst).returncode,0)
    def test_corrupt_archive_fails_before_extracting(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=self.fixture(p);archive=src/'release/tiny.tar.gz'
            with archive.open('ab') as f:f.write(b'corruption')
            dst=p/'restored';self.assertNotEqual(self.run_restore(src,dst).returncode,0)
            self.assertFalse((dst/'artifacts/记录.txt').exists())
    def test_existing_different_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=self.fixture(p);dst=p/'restored';(dst/'artifacts').mkdir(parents=True)
            file=dst/'artifacts/记录.txt';file.write_text('keep me',encoding='utf-8')
            self.assertNotEqual(self.run_restore(src,dst).returncode,0)
            self.assertEqual(file.read_text(encoding='utf-8'),'keep me')
    def test_path_traversal_rejected_even_with_valid_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=self.fixture(p);out=src/'release'
            data=b'escape';name='../escape.txt';archive=out/'tiny.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
            manifest=out/'tiny.manifest.json';manifest.write_text(json.dumps({'files':[{'path':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}))
            inv=out/'tiny.release.json';info=json.loads(inv.read_text())
            info['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
            info['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
            info['parts']=[{'name':archive.name,'bytes':archive.stat().st_size,'sha256':info['archive_sha256']}]
            inv.write_text(json.dumps(info));self.assertNotEqual(self.run_restore(src,p/'restored').returncode,0)
            self.assertFalse((p/'escape.txt').exists())

if __name__=='__main__':unittest.main()
