load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_binary (
    name = "libtfdecoder.so",
    srcs = ["tfdecoder.cc", "tfdecoder.h"],
    visibility=["//visibility:public"],
    linkshared = 1,
    deps = [
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/c:c_api",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/contrib/rnn:lstm_ops_kernels",
        "//tensorflow/contrib/rnn:lstm_ops_op_lib",
    ],
)

cc_library (
    name = "decoderlib",
    hdrs = ["tfdecoder.h"],
    srcs = ["libtfdecoder.so"],
    visibility = ["//visibility:public"],
)

cc_binary (
    name = "tacotron",
    srcs = ["main.cc", "tfdecoder.h"],
    deps = [":decoderlib"],
)
