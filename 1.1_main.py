from funasr import AutoModel

model = AutoModel(model="paraformer-zh")

res = model.generate(input=r"C:\Users\zljlld\Desktop\Theresia\ASR_test\test.wav")
print(res)