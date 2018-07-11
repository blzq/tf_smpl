# tf_smpl
TensorFlow interface for SMPL model. Code by https://github.com/akanazawa/hmr extracted as a separate repo.

Requires the SMPL model (`.pkl` file format):
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

The downloaded SMPL model contains `chumpy` arrays. `preprocess.py` (adapted from https://github.com/CalciferZh/SMPL) converts these to `numpy` arrays. Running it requires the `chumpy` library which is provided with the SMPL model (http://smpl.is.tue.mpg.de/).

### Citation
```
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```
