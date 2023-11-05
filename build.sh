#bazel build --copt=-mavx2 --copt=-mfma --copt=-mavx --copt=-msse4.2 --config=cuda  //tensorflow/tfdecoder:libtfdecoder.so --define framework_shared_object=false
bazel build -c opt --config=cuda  //tensorflow/tfdecoder:libtfdecoder.so --define framework_shared_object=false
