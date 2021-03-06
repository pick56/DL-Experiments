from keras.layers import Input,LSTM,Dense
from keras.models import Model,load_model
from keras.utils import plot_model

import pandas as pd
import numpy as np
import os
os.environ["PATH"] += os.pathsep + "E:/AAA/graphviz-2.38/release/bin"
#我的graphviz环境没配好，为了后面的Plot_model


def create_model(n_input, n_output, n_units):
    # 训练阶段,所有输入字符集大小、输出字符集大小、cell数量
    # encoder
    print(n_input)
    print(n_output)
    print(n_units)
    encoder_input = Input(shape=(None, n_input))
    # print(encoder_input)
    # encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    # n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    _, encoder_h, encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h, encoder_c]
    # 保留下来encoder的末状态作为decoder的初始状态

    # decoder
    decoder_input = Input(shape=(None, n_output))
    # decoder的输入维度为中文字符数
    decoder = LSTM(n_units, return_sequences=True, return_state=True)
    # 训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)
    # 在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    # 输出序列经过全连接层得到结果

    # 生成的训练模型
    model = Model([encoder_input, decoder_input], decoder_output)
    # 第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出

    # 推理阶段，用于预测过程
    # 推断模型—encoder
    encoder_infer = Model(encoder_input, encoder_state)

    # 推断模型-decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]  # 上个时刻的状态h,c

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,
                                                                                 initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]  # 当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)  # 当前时刻的输出
    decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)

    return model, encoder_infer, decoder_infer

N_UNITS = 300
BATCH_SIZE = 64
EPOCH = 50
NUM_SAMPLES = 1001


data_path = 'data/cmn.txt'



def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    state = encoder_inference.predict(source)
    #第一个字符'\t',为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,target_dict['\t']] = 1

    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':#预测到了终止符则停下来
            break
    return output


if __name__ == '__main__':
    print("helloworld")
    # 读取cmn-eng.txt文件
    df = pd.read_table(data_path, header=None).iloc[:NUM_SAMPLES, :, ]
    df.columns = ['inputs', 'targets']
    # 讲每句中文句首加上'\t'作为起始标志，句末加上'\n'作为终止标志
    df['targets'] = df['targets'].apply(lambda x: '\t' + x + '\n')

    input_texts = df.inputs.values.tolist()
    target_texts = df.targets.values.tolist()


    # 确定中英文各自包含的字符。df.unique()直接取sum可将unique数组中的各个句子拼接成一个长句子
    input_characters = sorted(list(set(df.inputs.unique().sum())))
    target_characters = sorted(list(set(df.targets.unique().sum())))
    # print(input_characters)
    # print(target_characters)
    # 1 / 0
    # INPUT_LENGTH，输入数据的时刻t的长度，这里为最长的英文句子长度
    # OUTPUT_LENGTH，输出数据的时刻t的长度，这里为最长的中文句子长度
    # INPUT_FEATURE_LENGTH，每个时刻进入encoder的lstm单元的数据xtxt的维度，这里为英文中出现的字符数
    # OUTPUT_FEATURE_LENGTH，每个时刻进入decoder的lstm单元的数据xtxt的维度，这里为中文中出现的字符数
    INUPT_LENGTH = max([len(i) for i in input_texts])
    OUTPUT_LENGTH = max([len(i) for i in target_texts])
    INPUT_FEATURE_LENGTH = len(input_characters)
    OUTPUT_FEATURE_LENGTH = len(target_characters)

    # encoder输入、decoder输入输出初始化为三维向量
    encoder_input = np.zeros((NUM_SAMPLES, INUPT_LENGTH, INPUT_FEATURE_LENGTH))
    decoder_input = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))
    decoder_output = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))
    print(encoder_input.shape)
    print(decoder_input.shape)
    print(decoder_output.shape)

    # 其中input_dict和target_dict为中英文字符与其索引的对应词典；input_dict_reverse和target_dict_reverse与之相反，索引为键字符为值：
    input_dict = {char: index for index, char in enumerate(input_characters)}
    input_dict_reverse = {index: char for index, char in enumerate(input_characters)}
    target_dict = {char: index for index, char in enumerate(target_characters)}
    target_dict_reverse = {index: char for index, char in enumerate(target_characters)}

    # 对句子进行字符级one - hot编码，将输入输出数据向量化
    # encoder的输入向量one-hot
    for seq_index, seq in enumerate(input_texts):
        for char_index, char in enumerate(seq):
            encoder_input[seq_index, char_index, input_dict[char]] = 1
    # decoder的输入输出向量one-hot，训练模型时decoder的输入要比输出晚一个时间步，这样才能对输出监督
    for seq_index, seq in enumerate(target_texts):
        for char_index, char in enumerate(seq):
            decoder_input[seq_index, char_index, target_dict[char]] = 1.0
            if char_index > 0:
                decoder_output[seq_index, char_index - 1, target_dict[char]] = 1.0
    # print(encoder_input)
    # print(decoder_output)
    # 输出一些结果
    print(''.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i) != 0]))
    print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i) != 0]))
    print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_input[0] if max(i) != 0]))

    # 创建模型
    model_train, encoder_infer, decoder_infer = create_model(INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, N_UNITS)
    # 查看模型结构
    plot_model(to_file='model.png', model=model_train, show_shapes=True)
    plot_model(to_file='encoder.png', model=encoder_infer, show_shapes=True)
    plot_model(to_file='decoder.png', model=decoder_infer, show_shapes=True)

    # In [17]:
    model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # In [18]:
    print(model_train.summary())

    print(encoder_infer.summary())

    print(decoder_infer.summary())

    model_train.fit([encoder_input, decoder_input], decoder_output, batch_size=BATCH_SIZE, epochs=EPOCH,
                    validation_split=0.2)

    for i in range(900, 1000):
        test = encoder_input[i:i + 1, :, :]  # i:i+1保持数组是三维
        out = predict_chinese(test, encoder_infer, decoder_infer, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH)
        # print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
        print(input_texts[i])
        print(out)

