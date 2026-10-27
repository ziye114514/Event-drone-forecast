"""One external input: a RAW file. No dataset manifests or labels are imported."""
from pathlib import Path

def open_raw(path):
    path=Path(path).resolve()
    if not path.is_file() or path.suffix.lower()!='.raw':
        raise ValueError('Expected an existing Prophesee RAW file')
    try:
        from metavision_core.event_io import RawReader
    except ImportError as e:
        raise RuntimeError('Use the Python version matching your installed Metavision SDK; this workstation uses Python 3.11.') from e
    return RawReader(str(path),do_time_shifting=True)

def raw_chunks(reader, start_s, duration_s, step_us):
    start_us=int(round(start_s*1e6))
    if start_us: reader.seek_time(start_us)
    end_us=start_us+int(round(duration_s*1e6)) if duration_s else 2**62
    while not reader.is_done() and reader.current_time<end_us:
        begin=max(start_us,int(reader.current_time))
        duration=min(step_us,end_us-begin)
        if duration<=0: break
        events=reader.load_delta_t(duration)
        # Each batch is [begin, end): the state is published at end and sees no later event.
        end=max(begin+duration,int(reader.current_time))
        events=events[(events['t']>=start_us)&(events['t']<min(end,end_us))]
        yield min(end,end_us)/1e6,events
