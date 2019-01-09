import matplotlib.pyplot as plt


fo = open("loss_up.txt", "r")
i=0
train=[]
test=[]
for line in fo:
    if(line == '\n'):
        continue
    i+=1
    if(  i == 61 ):
        test.append(eval(line))
        i = 0
    else:
        train.append(eval(line))

y_train = list(range(3060))
y_test = list(range(51))
y_test = [60*i for i in y_test]

plt.plot(y_train[3:300:2],train[3:300:2],label='value1')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()