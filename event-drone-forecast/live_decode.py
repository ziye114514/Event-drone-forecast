"""Official SDK delta-time iterator for paced replay, keeping native timestamps."""
def native_batches(raw, begin, duration,build_index=True,step_us=20000):
    from metavision_core.event_io import EventsIterator
    if not isinstance(step_us,int) or step_us<=0:raise ValueError('Positive integer step_us required')
    options=dict(start_ts=round(begin*1e6),delta_t=step_us,max_duration=round(duration*1e6),
                 mode='delta_t',relative_timestamps=False)
    if build_index:
        iterator=EventsIterator(input_path=str(raw),**options)
    else:
        # Match installed raw_reader.initiate_device's time shift and read
        # chunk. Its sequential seek does not use HAL's background index.
        # Public SDK configuration only; never patch the reader or decoder.
        from metavision_hal import DeviceDiscovery,RawFileConfig
        config=RawFileConfig();config.do_time_shifting=True
        config.n_events_to_read=100000;config.build_index=False
        device=DeviceDiscovery.open_raw_file(str(raw),config)
        if device is None:raise OSError('Cannot open RAW with explicit non-indexed configuration')
        iterator=EventsIterator.from_device(device,**options)
    if tuple(iterator.get_size())!=(720,1280):raise ValueError('Expected1280x720 RAW')
    for events in iterator:
        t=iterator.get_current_time()/1e6
        assert not len(events) or (events['t'].min()>=round(t*1e6)-step_us and events['t'].max()<round(t*1e6))
        yield t,events
