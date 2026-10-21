# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""A2 910B3 / CANN 9.0 resident-data, measured single-duration tile costs.

Generated from 206 numerically checked operation/shape observations on NPU6.
Counter: GetSystemCycle, 50 MHz; 3 warmups and 5 samples per count.
These are two-slot incremental helper costs, not bare instruction latency.
Errors below are held-out-shape residuals, NOT whole-kernel accuracy promises.
A3 has no hardware calibration. Missing families retain a labeled fallback.
"""

from functools import lru_cache

SOURCE_REPORT_SHA256 = "bf2297de112dd6a831e72cf2873d979f1141f4013c647e21db8343cea3056e31"
EVIDENCE = "sha256:ca33ce9987ce6cee177a5f785743c4e296c30de628a4db4f0faf37c4889bd9c1"
INPUT_SHA256 = (
    ("linear-v1/observations.json",
     "337fa8673a4b51ff4a1e0656d037be1711f3e237d1429a9e61cf1c01ddf2da1b"),
    ("linear-reduce-v2/observations.json",
     "7be0462b6a2653ab42548cc45fa9972c85f985e801b00f6a69438f1e06676335"),
    ("linear-copy-v1/observations.json",
     "964e30ae93829088c4f975f5f629146c767dc9fc1f2902270ac1724f0b0e9530"),
    ("linear-vector-v2/observations.json",
     "61fd07121cca3a4db1ece4dcad8d872407aa1c8d76ef6943298e984185e65745"),
    ("linear-cube-copy-v1/observations.json",
     "6b916eaa2d198d7e05e94a1b0dd907713a2e75665744bcb397b0fb127350af30"),
    ("linear-fa-cube-v2/observations.json",
     "585db4435374597abd06664daace505d8b73341577cddcfed5b213f5b2eff5c8"),
    ("linear-small-v1/observations.json",
     "e0eb1fdf816bd1b43b43327da7b2d08419e389a1ab31f41dd49a1e470cfa4e76"),
    ("linear-copy-fp16-v1/observations.json",
     "cc56e0daa33bc7e0f525b58a27570f309bb2c47d3e93d16a0663f541aad52f36"),
    ("linear-row-helpers-v1/observations.json",
     "dedde87ddffc739c42433cdcf985dca25b77d138123cfe7fea36f7100ab44df4"),
    ("linear-row-helpers-v2/observations.json",
     "8f078cff1e430d7835a8d33ecf2e93cc2042014a578f95bf273738c3e861d6f6"),
    ("linear-cast-v1/observations.json",
     "36f2cf61acad97352a33cc7c0766522d4b3341653bb0b987b3ebc8697b0a75f5"),
    ("linear-muls-v1/observations.json",
     "1fb8055f0d944a6b3fcee61a1e13b5c4b1156a799bd9280adf43982771e48803"),
    ("linear-fa-extra-v1/observations.json",
     "ba602994eff6c263c6a2554f5d28cd645fe0c3924145df020d303593608c7754"),
)

# op, dtype, geometry, ns/feature, intercept_ns, min/max feature,
# feature kind, maximum held-out absolute relative error, held-out shape count.
COSTS = (
    ("add", "float32", "contiguous", 0.004606961291497232, 11.751184618188432, 8192, 49152, "bytes",
     0.0005474814940323869, 3),
    ("axpy", "float16_to_float32", "contiguous", 0.006774061717916048, 20.601340180686805, 4096,
     24576, "bytes", 0.1349558394835888, 1),
    ("axpy", "float32", "contiguous", 0.004311280291642054, 16.968825665859583, 8192, 49152,
     "bytes", 0.012521666836602607, 1),
    ("brcb_experiment", "float32", "contiguous", 0.01758164696453617, 9.434941227542911, 32, 256,
     "bytes", 0.005919754865917781, 1),
    ("cast", "float16_to_float32", "contiguous", 0.004569635677022312, 17.131759483454402, 4096,
     24576, "bytes", 0.16509489753128115, 1),
    ("cast", "float32_to_float16", "contiguous", 0.0013609160181166848, 16.595312947486264, 8192,
     49152, "bytes", 0.1260852353992129, 1),
    ("copy_gm_l1", "float16", "contiguous", 0.0030852479115955084, 46.34801960658818, 8192, 65536,
     "bytes", 0.04585522699088451, 1),
    ("copy_gm_ub", "float16", "contiguous", 0.006523093590277915, 1.4386276653908965, 4096, 24576,
     "bytes", 0.007720620974205339, 1),
    ("copy_gm_ub", "float32", "contiguous", 0.006589155816141724, -1.4991928974979771, 8192, 49152,
     "bytes", 0.014152440883344886, 1),
    ("copy_l0c_gm", "float32", "row_elements_128", 0.0047747001120571715, 141.89516129032253, 16384,
     65536, "bytes", 0.0025369635156407, 1),
    ("copy_l0c_gm", "float32_to_float16", "row_elements_128", 0.0022641019337737616,
     145.0384024577573, 16384, 65536, "bytes", 0.02608214705931089, 1),
    ("copy_l0c_gm", "float32_to_float16", "row_elements_256_stride_elements_512",
     0.002332397320303499, 148.5176651305684, 32768, 131072, "bytes", 0.019839334504563877, 1),
    ("copy_l1_l0a", "float16", "contiguous", 0.0023021134810961236, -0.35511606076123314, 8192,
     65536, "bytes", 0.007450094161958631, 1),
    ("copy_l1_l0b", "float16", "contiguous", 0.004446625729880045, 3.03774108209592, 8192, 32768,
     "bytes", 0.01110093205576601, 1),
    ("copy_l1_l0b_transpose", "float16", "contiguous", 0.0043416171492118355, 0.5062724014336837,
     2048, 32768, "bytes", 0.0004657286210685285, 1),
    ("copy_ub_gm", "float16", "contiguous", 0.010188222226307232, -1.6123174256034076, 4096, 24576,
     "bytes", 0.01800774725304566, 1),
    ("copy_ub_gm", "float32", "contiguous", 0.009959592657574153, 1.2370147621650744, 8192, 49152,
     "bytes", 0.008988713801891368, 1),
    ("copy_ub_ub", "float32", "contiguous", 0.0020394170863082674, 9.766235883329589, 64, 2048,
     "bytes", 0.027944464879938957, 2),
    ("div", "float32", "contiguous", 0.008682210852172504, 16.11152334088368, 8192, 49152, "bytes",
     0.00005381583710029059, 1),
    ("exp", "float32", "contiguous", 0.004339111383337399, 15.648968797070744, 64, 49152, "bytes",
     0.0025895926951395865, 3),
    ("fill", "float32", "contiguous", 0.002166558927274689, 9.63546149651011, 64, 49152, "bytes",
     0.014917648316997267, 3),
    ("max", "float32", "contiguous", 0.00428632664357344, 11.244075625596281, 64, 2048, "bytes",
     0.0335914172709888, 2),
    ("mma", "float16", "contiguous", 0.00013558304941848882, 0.13609420621244794, 262144, 4194304,
     "padded_mnk", 0.0011152453594296396, 3),
    ("mma_init", "float16", "contiguous", 0.00013560303894605505, 0.13920890937021113, 524288,
     2097152, "padded_mnk", 0.00018077815552319798, 1),
    ("mul", "float32", "contiguous", 0.004386521468306174, 16.452979770366326, 8192, 49152, "bytes",
     0.05673648829766076, 1),
    ("mul_add_dst", "float32", "contiguous", 0.013335544319069473, 16.117724233518608, 64, 2048,
     "bytes", 0.03437762457266258, 2),
    ("muls", "float32", "contiguous", 0.0022859831373134368, 12.813432249197934, 64, 49152, "bytes",
     0.010349931988098193, 3),
    ("reduce_max", "float32", "last_axis_width_512", 0.002686824499140826, 50.36498607097295, 8192,
     49152, "bytes", 0.12020438904616593, 1),
    ("reduce_sum", "float32", "last_axis_width_512", 0.0023983639031287837, 66.98817334478898, 8192,
     49152, "bytes", 0.12609302244171905, 1),
    ("row_expand_div_experiment", "float32", "row_stride_bytes_512", 0.006696438616714329,
     17.559043778801822, 4096, 16384, "bytes", 0.07881210519922255, 1),
    ("row_expand_mul_experiment", "float32", "row_stride_bytes_512", 0.004075489625978919,
     18.3126920122888, 4096, 16384, "bytes", 0.10785937919312993, 1),
    ("row_expand_sub_experiment", "float32", "row_stride_bytes_2048", 0.004697443493526442,
     11.119431643625195, 2048, 8192, "bytes", 0.04330989066051072, 1),
    ("sub", "float32", "contiguous", 0.004597377880253467, 12.085239157274017, 64, 49152, "bytes",
     0.020679331981310112, 3),
)


def calibration_report():
    families = []
    for op, dtype, geometry, a, b, low, high, feature, error, count in COSTS:
        families.append({
            "operation": op,
            "dtype": dtype,
            "geometry": geometry,
            "measurement_mode": "two_slot",
            "status": "fitted",
            "cost": {
                "slope_ns": a,
                "intercept_ns": b,
                "min_feature": low,
                "max_feature": high,
                "feature": feature,
                "evidence": EVIDENCE
            },
            "held_out_count": count,
            "held_out_max_abs_relative_error": error,
        })
    return {
        "schema": "tilelang-linear-cost-calibration-v1",
        "unit": "ns",
        "platform_measured": "A2",
        "a3_validated": False,
        "source_report_sha256": SOURCE_REPORT_SHA256,
        "measurement_scope": "910B3 CANN9, resident-data single-core probes",
        "families": families
    }


@lru_cache(maxsize=1)
def default_a2_timing_profile():
    from .linear_timing import measured_timing_profile
    from .profile import TimingProfile

    measured = measured_timing_profile(calibration_report())
    fallback = TimingProfile(
        "A2", estimator="pypto-fast", calibration="pypto-fast-derived-uncalibrated-49587f3")

    class MeasuredWithFallback(type(measured)):

        def estimate_task_with_source(self, operation, *, pipe, metadata):
            try:
                return super().estimate_task_with_source(operation, pipe=pipe, metadata=metadata)
            except ValueError:
                return (fallback.estimate_task(operation, pipe=pipe,
                                               metadata=metadata), "pypto-fallback")

    return MeasuredWithFallback(
        "A2",
        estimator="measured-linear",
        calibration="a2-910b3-cann9-measured-linear+pypto-fallback-" + SOURCE_REPORT_SHA256[:16])
