# Dogs Breed classification 

## Example work 

### 1st Download weight

```
./download_weights.sh
```

### Simple flask app
<img src="images/example 1.png">

Run:
```bash
python app.py
```

Or:
```bash
docker build -t dogs-breed  .                                                                                              ✔ 
docker run -it -d -p 5000:5000 dogs-breed
```

then go "http://localhost:5000/'

### Simple telegram bot

<img src="images/example 2.png">

**Usage**:  
Insert your secret toke in 1st line of ```telegram_bot.py``` here:
```
secret_key = "your secret token"

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
...
```

then run:
```
python telegram_bot.py
```
