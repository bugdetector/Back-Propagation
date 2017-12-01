import numpy as np
from sklearn.preprocessing import LabelBinarizer
def load_data():
    from sklearn.datasets import load_digits
    digits = load_digits()
    return digits.images, digits.data, digits.target

def show_image(images, index):
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(1, 1))
    plt.imshow(images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

def egit(epoch, eta, egitim, target_output): #egitim fonksiyonu
    np.random.seed(1)
    layer_1_w = np.random.rand(66, 6) # katmanlar için ağırlıklar rastgele değerler ile oluşturuluyor
    layer_2_w = np.random.rand(6, 4)
    layer_3_w = np.random.rand(4, 5)
    layer_4_w = np.random.rand(5,10)

    for epoch in range(0, epoch):
        for x, t in zip(egitim, target_output):
            x = x[np.newaxis]

            layer_1_o = sig(np.dot(x, layer_1_w))
            layer_2_o = sig(np.dot(layer_1_o, layer_2_w))
            layer_3_o = sig(np.dot(layer_2_o, layer_3_w))
            result = sig(np.dot(layer_3_o, layer_4_w))

            layer_4_delta = ((-(t - result)) * ((1 - result) * result))
            layer_3_delta = layer_4_delta.dot(layer_4_w.T) * ((1 - layer_3_o) * layer_3_o)
            layer_2_delta = layer_3_delta.dot(layer_3_w.T) * ((1 - layer_2_o) * layer_2_o)
            layer_1_delta = layer_2_delta.dot(layer_2_w.T) * ((1 - layer_1_o) * layer_1_o)

            layer_4_w -= eta * (layer_4_delta.T * layer_3_o).T
            layer_3_w -= eta * (layer_3_delta.T * layer_2_o).T
            layer_2_w -= eta * (layer_2_delta.T * layer_1_o).T
            layer_1_w -= eta * (layer_1_delta.T * x).T
    return layer_1_w, layer_2_w, layer_3_w, layer_4_w

def test_et(layer_1_w,layer_2_w,layer_3_w,layer_4_w,test,test_etiket): #test fonksiyonu
    sonuclar = []
    for x, t in zip(test, test_etiket):
        layer_1_o = sig(np.dot(x, layer_1_w))
        layer_2_o = sig(np.dot(layer_1_o, layer_2_w))
        layer_3_o = sig(np.dot(layer_2_o, layer_3_w))
        result = sig(np.dot(layer_3_o, layer_4_w))
        sonuclar.append(result)
    sonuclar = np.array(sonuclar)
    sonuclar = label_binarizer.inverse_transform(sonuclar)

    dogru_sayisi = 0;
    for res, t in zip(sonuclar,test_etiket):
        if(res == t):
            dogru_sayisi+=1
    return 100*dogru_sayisi/len(test) #testin doğruluk oranı hesaplanıyor

sig = lambda t: 1/(1+np.exp(-t))


images, data, target = load_data()
data = data/64

label_binarizer = LabelBinarizer()
label_binarizer.fit(target)
target_output = label_binarizer.transform(target)

egitim_verisi = []
test_verisi  = []
#eğitim verisi oluşturuluyor
for i in range(0,int(len(images))):
    avg = np.average(data[i]) #kendi seçtiğim veri olarak dizinin ortalamasını ekledim
    singleData = []
    singleData.extend(data[i])
    singleData.append(avg)
    singleData.append(1) #bias ekleniyor
    egitim_verisi.append(singleData)
#test verisi oluşturuluyor
for i in range(int(len(images)/2),len(images)):
    value = np.average(data[i])
    singleData = []
    singleData.extend(data[i])
    singleData.append(value)
    singleData.append(1)
    test_verisi.append(singleData)
#eğitimde kullanılacak etiketler
egitim_etiket = []
egitim_etiket.extend(target_output[0:int(len(target_output) / 2)])
#testte kullanılacak etiketler
test_etiket = []
test_etiket.extend(target[int(len(target) / 2):len(target)])
egitim_verisi = np.array(egitim_verisi) # eğitim verisi numpy dizisi olmalı

epoch = 1000


print("eta,accuracy")
eta = 0.1
while(eta<=1): # eta 0.1'er artırılarak test yapılıyor ve doğruluk oranı yazdırılıyor
    layer_1_w,layer_2_w,layer_3_w,layer_4_w = egit(1400,eta,egitim_verisi,egitim_etiket)
    accuracy = test_et(layer_1_w,layer_2_w,layer_3_w,layer_4_w,test_verisi,test_etiket)
    print(str(eta) +","+ str(accuracy))
    eta += 0.1