# Emotion Recognition

This project trains FER2013 emotion-recognition models for classroom engagement analysis.

## Recommended Notebook

Use the top-level notebook:

- [emotion_recognition.ipynb](/Users/gehanpasindhu/Documents/projects/datascience/diyasha/emotion_recognition.ipynb)

That notebook has been updated to:

- fix the broken dataset-validation cell
- auto-detect the dataset in this workspace
- include the missing `scipy` dependency
- handle zero-division cases in evaluation metrics more safely

## Dataset Layout In This Workspace

The available FER2013-style folders are:

- `archive/train`
- `archive/test`

The updated notebook now checks these candidate locations automatically:

- `./train` and `./test`
- `./archive/train` and `./archive/test`
- `../archive/train` and `../archive/test`

## Install

```bash
pip install -r requirements.txt
```

You can also run the notebook install cell directly.

## Notes

- `emotion-rec-1/emotion_recognition.ipynb` is a duplicated notebook copy.
- `emotion-rec-1/FER2013_emotion_model.ipynb` is an older experimental notebook and does not represent the cleanest training/evaluation path.
