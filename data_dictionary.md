# Data Dictionary

Source file: `data/raw/labeled_data.csv`

| Column | Meaning |
| --- | --- |
| `count` | Total annotation count for the tweet |
| `hate_speech` | Number of annotators selecting hate speech |
| `offensive_language` | Number of annotators selecting offensive language |
| `neither` | Number of annotators selecting neither |
| `class` | Target label: `0` hate speech, `1` offensive language, `2` neither |
| `tweet` | Original tweet text |

The processed file keeps `tweet` and `class`, and adds `clean_tweet`. The raw file is preserved as the source of truth and must not be edited.
