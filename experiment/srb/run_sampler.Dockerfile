FROM srb

WORKDIR /surface-reconstruction-benchmark

ADD run_sampler_fix.diff /run_sampler_fix.diff

RUN git apply ../run_sampler_fix.diff

ENTRYPOINT ["python3", "scripts/RunSampler.py" ]
