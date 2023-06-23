import matplotlib.pyplot as plt

## epoch 마다의 성능 선그래프
total_f1 = [0.9210610418664111, 0.9031446540880502, 0.9189704480457578, 0.9240628003844923, 0.9168525007964319, 0.9205741626794259, 0.9220904135941007, 0.9329501915708812, 0.9331627758234731, 0.9302176696542893, 0.9312440038375439, 0.9337579617834395, 0.936684696150175, 0.9392300349984092, 0.9392300349984092, 0.9399428026692087, 0.9381933438985737, 0.9392020265991134, 0.9382716049382716, 0.9383106611831699, 0.9358609794628752]
length_f1 = [0.00, 0.01, 0.02, 0.04, 0.05, 0.19 ,0.25,  0.92, 0.95, 0.92, 0.94, 0.95, 0.96 , 0.95 , 0.95 , 0.95 , 0.92, 0.94, 0.94, 0.92, 0.90]
epoch = list(range(0, 21, 1))
xe = list(range(0, 21, 5))
plt.plot(epoch, length_f1, color = 'indianred', linewidth=2.0)
plt.plot(epoch, total_f1, color = 'steelblue', linewidth=2.0)
plt.xticks(xe)

plt.xlabel('epochs')
plt.ylabel('F1')
plt.title('<F1 score of EWC NER model>', fontsize=20)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.legend(loc = 'lower right')

# annotate point
x, y = 16, 0.9399428026692087
plt.plot([x, x], [0, y], linewidth=2.5, linestyle='--')
plt.scatter([x,], [y,], 50)
plt.annotate('Max:0.9399', xy = (x, y),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
)

plt.savefig('graph1')

# ## 성능 비교 막대 그래프 
# index = ['original NER', 'Rehearsal', 'EWC']
# compare = [0.9354, 0.9269, 0.9399]
# plt.bar(index, compare,color = 'lightblue', width = 0.4)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# for i, v in enumerate(index):
#     plt.text(v, compare[i], compare[i],
#              fontsize = 9,
#              color='black',
#               horizontalalignment='center',
#               verticalalignment='bottom')

# plt.ylim(0.92, 0.95)
# plt.xlabel('Model')
# plt.ylabel('F1 score')
# plt.savefig('graph2')