To setup follow these steps:

## 1: Create a virtual environment

```
python -m venv ./venv
```

## 2: Activate venv

You need to use bash to run this (not the terminal). Bash should be included with git, if you don't have git, install git. Make sure that you run this command from this directory. Do either of these things:

1. **Open git bash with context menu:** If you are on windows, you can go to the folder that this is in and `shift` + `right click` anywhere in the folder to open a menu, if you have git bash installed correctly there should be an option that says 'Git bash here'. Click that
2. **Use terminal navigation:** First open git bash however you would with any program. I believe on mac you can use spotlight, and on windows just press windows key and type "git bash". To get to the directory, use `ls` to list the folders from where you are, then use `cd <folder>` to get to the directory. For example mine opens to my user path, so I use `cd desktop` then `cd chess-ml`. Once you are at the root of this project, `ls` should return all the files at the base, like `main.py` and `model.py`.

Once your git bash terminal is at the root of this project, you can run:

```bash
source ./venv/Scripts/activate
```

## 3: Install packages

To install packages, in your terminal (not git bash), run

```
pip install -r requirements.txt
```

## 4: Download data

Download whatever dataset you want from lichess. I recommend the smallest one possible as it still contains more than 100k games, once I have this more dialed in im gonna train it with ~1m games, but I expect that to take a few hours so ima wait on that. Make sure that you decompress it and that the data file is called `data.pgn` and is in the root of this project

## 5: Run script

Hopefully that works.
To run this project, just use

```
python main.py
```

It may be extremely slow. So you can adjust line 368 in `main.py`:

```py
train_df, test_df = prepare_training_data("data.pgn", num_games=20000)
```

Adjust num_games to something lower, do know though that this will make the algorithm perform far worse. On my computer with an i7-2700kf, 32gb of ram, and a 1070 run 20,000 games in about 1 - 1.5 mins.
