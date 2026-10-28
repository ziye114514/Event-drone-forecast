"""Build a private SDK adapter; no vendor headers/DLLs copied into the project."""
from pathlib import Path
import argparse,sys,sysconfig,subprocess,os,json,hashlib,time,shutil
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root',required=True,type=Path,help='Installed SDK root with include/ and third_party/include/')
    p.add_argument('--pybind-include',type=Path,help='Existing pybind11 include directory; otherwise import installed pybind11')
    p.add_argument('--vswhere',type=Path,help='Optional Visual Studio vswhere.exe path')
    p.add_argument('--output',type=Path,default=ROOT/'.cache/sdk_activity_avx2')
    a=p.parse_args()
    if os.name!='nt' or sys.maxsize<=2**32:raise RuntimeError('64-bit Windows/MSVC is required for this optional adapter')
    sdk=a.sdk_root.expanduser().resolve()
    if not (sdk/'include/metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h').is_file():raise FileNotFoundError('Installed SDK CV headers not found')
    if a.pybind_include:pybind=a.pybind_include.expanduser().resolve()
    else:
        try:
            import pybind11
            pybind=Path(pybind11.get_include()).resolve()
        except ImportError as e:raise RuntimeError('Provide --pybind-include or install pybind11 in the build environment') from e
    if not (pybind/'pybind11/pybind11.h').is_file():raise FileNotFoundError('pybind11 headers not found')
    tool=a.vswhere or Path(os.environ.get('ProgramFiles(x86)',os.environ.get('ProgramFiles','')))/'Microsoft Visual Studio/Installer/vswhere.exe'
    if not tool.is_file():raise FileNotFoundError('MSVC tools or --vswhere are required')
    install=subprocess.check_output([str(tool),'-utf8','-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True,encoding='utf-8').strip()
    if not install:raise RuntimeError('No suitable MSVC C++ installation found')
    dev=Path(install)/'Common7/Tools/VsDevCmd.bat'
    # A fixed compiler environment initialization, with no file mutations in
    # another shell. Actual compilation uses an argv list, never shell text.
    if any(c in str(dev) for c in '\"&|<>^'):raise ValueError('Unexpected compiler initialization path')
    result=subprocess.run(f'cmd.exe /d /u /s /c ""{dev}" -arch=x64 -host_arch=x64 >nul && set"',capture_output=True,text=True,encoding='utf-16-le',errors='replace')
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    env={k.upper():v for k,v in os.environ.items()}
    for line in result.stdout.splitlines():
        if '=' in line and not line.startswith('='):
            k,v=line.split('=',1);env[k.upper()]=v
    # These are existing headers, not copied SDK sources.
    out=a.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'native/sdk_activity_avx2.cpp';target=out/('_edf_sdk_activity'+sysconfig.get_config_var('EXT_SUFFIX'))
    compiler=shutil.which('cl.exe',path=env['PATH'])
    if compiler is None:raise RuntimeError('Compiler missing from initialized VS environment')
    args=[compiler,'/nologo','/O2','/LD','/EHsc','/std:c++17','/MD','/arch:AVX2','/utf-8',
        '/I'+sysconfig.get_path('include'),'/I'+str(pybind),'/I'+str(sdk/'include'),'/I'+str(sdk/'third_party/include'),str(source),
        '/Fo'+str(out/'sdk_activity_avx2.obj'),'/link','/LIBPATH:'+str(Path(sys.base_prefix)/'libs'),'/OUT:'+str(target)]
    start=time.perf_counter();build=subprocess.run(args,cwd=out,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (out/'build.log').write_text(build.stdout+'\n'+build.stderr,'utf-8')
    if build.returncode:raise RuntimeError('Build failed; see '+str(out/'build.log'))
    data=dict(extension=str(target),build_s=time.perf_counter()-start,compiler=install,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        vendor_headers_copied_or_modified=False,private_binary_not_for_release=True,requires_valid_installed_SDK_license=True)
    (out/'build.json').write_text(json.dumps(data,indent=2),'utf-8');print(json.dumps(data))


if __name__=='__main__':main()
