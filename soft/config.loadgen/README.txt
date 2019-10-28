
Installing/detecting variations of this soft (depends on an MLPerf inference git checkout) :


    ck detect soft --tags=config,loadgen,test01

    ck detect soft --tags=config,loadgen,test04a

    ck detect soft --tags=config,loadgen,test04b

    ck detect soft --tags=config,loadgen,test05

    ck detect soft --tags=config,loadgen,original.mlperf.conf


UPDATE: If your ck-env is sufficiently fresh (ck pull repo:ck-env), you can also fix the source git repository:

    ck detect soft --tags=config,loadgen,test04b --dep_add_tags.mlperf-inference-src=upstream.master --extra_tags=from-master

    ck detect soft --tags=config,loadgen,test04a --dep_add_tags.mlperf-inference-src=upstream.pr518 --extra_tags=from-pr518
