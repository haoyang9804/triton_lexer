def has_warp_spec():
    import triton.language as tl

    return hasattr(tl, "async_task")
