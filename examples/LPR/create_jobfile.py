import os
### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()

solver_file = './jobs/Resnet_LSTM_CTC/solver.prototxt'
job_dir = './jobs/Resnet_LSTM_CTC/'
job_file = "{}/train.sh".format(job_dir)

# Defining which GPUs to use.
gpus = "0"
model_name = "lpr_resnet_lstm"

snapshot_dir = './models/LPR'
remove_old_models = False
resume_training = False
pretrain_model = './models/LPR/lpr_resnet_lstm_iter_12000.caffemodel'
snapshot_prefix = 'lpr_resnet_lstm'
max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    #iter = int(basename.split("{}_iter_".format(snapshot_prefix))[1])
    iter = int(basename.split("lpr_resnet_lstm_iter_")[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))


# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  #f.write(train_src_param)
  f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
