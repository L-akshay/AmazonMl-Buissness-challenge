"""Stream validated private Kaggle deliverables to disk without loading them into RAM."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def destination(folder,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe remote output path')
    target=(folder/relative).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise ValueError('Output escapes delivery directory')
    return target


def stream_file(url,target,expected_sha=None):
    import requests
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and expected_sha:
        with target.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()==expected_sha:
                return {'bytes':target.stat().st_size,'sha256':expected_sha,'reused':True}
    part=target.with_suffix(target.suffix+'.part')
    offset=part.stat().st_size if part.exists() else 0
    response=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120))
    response.raise_for_status()
    if response.status_code==206:
        if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
            raise ValueError('Unexpected download resume offset')
    else:
        offset=0
    remaining=int(response.headers.get('Content-Length','0'))
    if shutil.disk_usage(target.parent).free<remaining+2*1024**3:
        raise OSError('Insufficient disk headroom for deliverable')
    checksum=hashlib.sha256()
    if offset:
        with part.open('rb') as f:
            while chunk:=f.read(1024**2):
                checksum.update(chunk)
    received=0
    try:
        with part.open('ab' if offset else 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024**2):
                f.write(chunk); checksum.update(chunk); received+=len(chunk)
    finally:
        response.close()
    if remaining and received!=remaining:
        raise ValueError('Truncated download; resumable .part retained')
    sha=checksum.hexdigest()
    if expected_sha and sha!=expected_sha:
        raise ValueError('Downloaded TSV differs from remotely validated hash')
    part.replace(target)
    return {'bytes':offset+received,'sha256':sha,'reused':False}


def download(kernel,folder,package=False):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api=KaggleApi(); api.authenticate()
    owner,slug=kernel.split('/')
    files=[]; token=None
    with api.build_kaggle_client() as client:
        while True:
            request=ApiListKernelSessionOutputRequest()
            request.user_name=owner; request.kernel_slug=slug
            request.page_size=200
            if token:
                request.page_token=token
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            token=response.next_page_token
            if not token:
                break
    reports={}; validation=None
    if not package:
        checks=[f for f in files if f.file_name.endswith('output/cloud/validation.json')]
        if len(checks)!=1:
            raise ValueError('No unique completed TSV validation report')
        target=destination(folder,checks[0].file_name)
        stream_file(checks[0].url,target)
        validation=json.loads(target.read_text())
    selected=[f for f in files if (f.file_name.endswith('output/submission_package.zip') if package
               else '/output/cloud/' in '/'+f.file_name and f.file_name.endswith('/matching_results.tsv'))]
    if not selected:
        raise ValueError('No completed deliverables found')
    for file in selected:
        expected=None
        if validation is not None:
            variant=Path(file.file_name).parent.name
            if validation[variant]['organizer']!='PASS':
                raise ValueError('Unvalidated variant')
            expected=validation[variant]['sha256']
        reports[file.file_name]=stream_file(file.url,destination(folder,file.file_name),expected)
        print(json.dumps({'file':file.file_name,**reports[file.file_name]}),flush=True)
    (folder/'download_validation.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--package',action='store_true')
    args=parser.parse_args()
    download(args.kernel,args.output,args.package)
