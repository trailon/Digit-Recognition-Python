from tkinter import *
from tkinter.filedialog import askopenfilename
from functools import partial
import numpy as np
from PIL import Image, ImageTk
from numpy import asarray
import cv2  #version 3.2.0
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

#sayigenisligi = 10
#sayiyuksekligi = 20
resimyukseklıgı = 28
resimgenisligi = 28

def imresize(arr,size):
	img=Image.fromarray(arr)
	img=img.resize(size)
	return asarray(img)


def GUI():
    root = Tk()
    root.geometry("1600x800")
    root['background'] = '#5395b5'
    root.title("Sayı Tanıma")
    dosyayoludegiskeni = StringVar()
    dosyayolugirisi = Entry(root, width=100, textvariable=dosyayoludegiskeni)
    dosyayolugirisi.pack()
    dosyayolugirisi.focus()

    def dosyasecimi(dosyayoludegiskeni, event=None):
        dosyaismi = askopenfilename()
        print('Secilen Dosya', dosyaismi)
        dosyayoludegiskeni.set(dosyaismi)

        #------------------data hazirlama--------------------------------------------
        yuklenenresim = dosyaismi.split("images/")[1]
        sayiegitimiresmi = 'sayilar.png'
        kullanicisayiegitimi = 'ozelsayilar.jpg'
        kullaniciresmi = './images/' + yuklenenresim

        #sayilar, labels = sayilaryukle(sayiegitimiresmi) (iyi sonuc vermiyor)
        sayilar, labels = sayileryukleozel(
            kullanicisayiegitimi
        )  #elyazisi dataset dataset

        print('veri egitiliyor', sayilar.shape)
        print('veri test ediliyor', labels.shape)

        sayilar, labels = shuffle(sayilar, labels, random_state=256)
        sayiverisiniegit = pixels_to_hog_20(sayilar)
        X_train, X_test, y_train, y_test = train_test_split(sayiverisiniegit,
                                                            labels,
                                                            test_size=0.33,
                                                            random_state=42)

        #------------------egitim ve test----------------------------------------

        '''model = KNN_MODEL(k=3)
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        print('%lik başarı: ', accuracy_score(y_test, preds))

        model = KNN_MODEL(k=4)
        model.train(sayiverisiniegit, labels)
        proc_user_img(kullaniciresmi, model)'''

        model = SVM_MODEL(num_feats=sayiverisiniegit.shape[1])
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        print('%lik başarı: ', accuracy_score(y_test, preds))

        model = SVM_MODEL(num_feats=sayiverisiniegit.shape[1])
        model.train(sayiverisiniegit, labels)
        proc_user_img(kullaniciresmi, model)

        def goster():

            dosyayolu = './images/' + yuklenenresim
            image = Image.open(dosyayolu)
            image = image.resize((500, 500), Image.ANTIALIAS)
            img0 = ImageTk.PhotoImage(image)
            panel0 = Label(root, image=img0)
            panel0.photo = img0
            panel0.place(relx=0.013, rely=0.2)

            dosyayolu = "sonuc.jpg"
            image = Image.open(dosyayolu)
            image = image.resize((500, 500), Image.ANTIALIAS)
            img1 = ImageTk.PhotoImage(image)
            panel1 = Label(root, image=img1)
            panel1.photo = img1
            panel1.place(relx=0.342, rely=0.2)

            dosyayolu = "dijitalsonuc.jpg"
            image = Image.open(dosyayolu)
            image = image.resize((500, 500), Image.ANTIALIAS)
            img2 = ImageTk.PhotoImage(image)
            panel2 = Label(root, image=img2)
            panel2.photo = img2
            panel2.place(relx=0.671, rely=0.2)

        gosterbutton = Button(root, text='Göster', command=goster)
        gosterbutton.place(relx=0.485, rely=0.061)

    calistirbutton = Button(root,
                            text='Aç ve Çalıştır',
                            command=partial(dosyasecimi, dosyayoludegiskeni))
    calistirbutton.pack()

    root.mainloop()

'''def split2d(resim, hucreboyutu, duzhali=True):
    h, w = resim.shape[:2]
    sx, sy = hucreboyutu
    hucreler = [np.hsplit(row, w // sx) for row in np.vsplit(resim, h // sy)]
    hucreler = np.array(hucreler)
    if duzhali:
        hucreler = hucreler.reshape(-1, sy, sx)
    return hucreler'''


'''def sayileryukle(fn):
    print('Dosya "%s eğitim için yükleniyor" ...' % fn)
    sayiresmi = cv2.imread(fn, 0)
    sayilar = split2d(sayiresmi, (sayigenisligi, sayiyuksekligi))
    boyutlandirilmissayilar = []
    for digit in sayilar:
        boyutlandirilmissayilar.append(
            imresize(digit, (resimgenisligi, resimyukseklıgı)))
    labels = np.repeat(np.arange(nsinifi), len(sayilar) / nsinifi)
    return np.array(boyutlandirilmissayilar), labels'''


def pixels_to_hog_20(img_array):
    veriozellikleri = []
    for resim in img_array:
        fd = hog(resim,
                 orientations=10,
                 pixels_per_cell=(5, 5),
                 cells_per_block=(1, 1))
        veriozellikleri.append(fd)
    ozellikler = np.array(veriozellikleri, 'float64')
    return np.float32(ozellikler)


#ozel model tanimla
class KNN_MODEL():
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(
            samples, self.k)
        return results.ravel()


class SVM_MODEL():
    def __init__(self, num_feats, C=1, gamma=0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)  #SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1, self.features))
        return results[1].ravel()


def sayilarial(noktalar, hiyerarsi):
    hiyerarsi = hiyerarsi[0]
    ilksinirlayicilar = [cv2.boundingRect(ctr) for ctr in noktalar]
    sinirlayicilar = []
    #en yaygin hiyerarsi seviyesini bul, Bounding box lar burada.
    u, indisler = np.unique(hiyerarsi[:, -1], return_inverse=True)
    yayginhiyerarsi = u[np.argmax(np.bincount(indisler))]

    for r, hr in zip(ilksinirlayicilar, hiyerarsi):
        x, y, w, h = r
        # bu, tahmin etmeye çalıştığınız resme bağlı olarak değişebilir
        # YALNIZCA içinde resim bulunan dikdörtgenleri çıkarmaya çalışıyoruz (Böyle daha kolay)
        # diğer sayilar içindeki sayılardan gormezden gelmek için yalnızca aynı global katmandaki boxlari çıkarmak için hiyerarşi kullanıyoruz
        # Sayının görünümündeki döngüler nedeniyle her 6,9,8'in içinde bir box olabilir - bunu istemiyoruz.
        if ((w * h) > 400) and (10 <= w <= 200) and (
                10 <= h <= 200) and hr[3] == yayginhiyerarsi:
            sinirlayicilar.append(r)

    return sinirlayicilar


def proc_user_img(img_file, model):
    print('Dosya "%s sayı tanıma için yükleniyor" ...' % img_file)
    okunanresim = cv2.imread(img_file)
    bosresim = np.zeros((okunanresim.shape[0], okunanresim.shape[1], 3),
                        np.uint8)
    bosresim.fill(255)

    sayiresmiatama = cv2.cvtColor(okunanresim, cv2.COLOR_BGR2GRAY)
    plt.imshow(sayiresmiatama)
    kernel = np.ones((5, 5), np.uint8)

    ret, esikdegeri = cv2.threshold(sayiresmiatama, 127, 255, 0)
    esikdegeri = cv2.erode(esikdegeri, kernel, iterations=1)
    esikdegeri = cv2.dilate(esikdegeri, kernel, iterations=1)
    esikdegeri = cv2.erode(esikdegeri, kernel, iterations=1)

    noktalar, hiyerarsi = cv2.findContours(esikdegeri, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    sayidiktortgenleri = sayilarial(
        noktalar, hiyerarsi)  

    for dortgen in sayidiktortgenleri:
        x, y, w, h = dortgen
        cv2.rectangle(okunanresim, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sayiresmi = sayiresmiatama[y:y + h, x:x + w]
        sayiresmi = (255 - sayiresmi)
        sayiresmi = imresize(sayiresmi, (resimgenisligi, resimyukseklıgı))

        hog_img_data = pixels_to_hog_20([sayiresmi])
        pred = model.predict(hog_img_data)
        cv2.putText(okunanresim, str(int(pred[0])), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        cv2.putText(bosresim, str(int(pred[0])), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    plt.imshow(okunanresim)
    cv2.imwrite("sonuc.jpg", okunanresim)
    cv2.imwrite("dijitalsonuc.jpg", bosresim)
    #cv2.destroyAllWindows()


def get_contour_precedence(nokta, sutunlar):
    return nokta[1] * sutunlar + nokta[0]  #row-wise ordering



# bu fonksiyon, ozel egitim datalari işler
# bkz. örnek: custom_train.sayilar.jpg
# kendinizinkini kullanmak istiyorsanız, benzer bir biçimde olmalıdır
def sayileryukleozel(img_file):
    veriegit = []
    hedefegit = []
    baslatmadegeri = 1
    okunanresim = cv2.imread(img_file)
    sayiresmiatama = cv2.cvtColor(okunanresim, cv2.COLOR_BGR2GRAY)
    plt.imshow(sayiresmiatama)
    kernel = np.ones((5, 5), np.uint8)

    ret, esikdegeri = cv2.threshold(sayiresmiatama, 127, 255, 0)
    esikdegeri = cv2.erode(esikdegeri, kernel, iterations=1)
    esikdegeri = cv2.dilate(esikdegeri, kernel, iterations=1)
    esikdegeri = cv2.erode(esikdegeri, kernel, iterations=1)

    noktalar, hiyerarsi = cv2.findContours(esikdegeri, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    sayidiktortgenleri = sayilarial(
        noktalar, hiyerarsi)  

    #dikdortgenleri x ve y pozisyonuna gore sirala 
    sayidiktortgenleri.sort(
        key=lambda x: get_contour_precedence(x, okunanresim.shape[1]))

    for index, dortgen in enumerate(sayidiktortgenleri):
        x, y, w, h = dortgen
        cv2.rectangle(okunanresim, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sayiresmi = sayiresmiatama[y:y + h, x:x + w]
        sayiresmi = (255 - sayiresmi)

        sayiresmi = imresize(sayiresmi, (resimgenisligi, resimyukseklıgı))
        veriegit.append(sayiresmi)
        hedefegit.append(baslatmadegeri % 10)

        if index > 0 and (index + 1) % 10 == 0:
            baslatmadegeri += 1
    cv2.imwrite("veriegitimiresmi.jpg", okunanresim)

    return np.array(veriegit), np.array(hedefegit)


GUI()
