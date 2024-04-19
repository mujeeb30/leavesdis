import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
app=Flask(__name__)
model=load_model("classify_diseaseV2.h5")
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/result.html',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'static/assets/uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(256,256))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['Potato Early blight','Potato Late blight','Potato healthy','Tomato Early blight','Tomato Late blight','Tomato healthy']
        text=str(index[pred[0]])
    return render_template("result.html",path=f.filename,pred=text)

if __name__=='__main__':
    app.run(debug=True,port=3001)
    
