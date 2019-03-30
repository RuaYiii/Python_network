import numpy
class neuralNetwork:
    def __init__(self,inputnodes,hideennodes,outputnodes,learningrate):
        self.innodes= inputnodes
        self.hnodes= hideennodes
        self.onodes= outputnodes
        #lr =>learningrate 就是咱的学习率
        self.lr= learningrate
        self.wih= numpy.random.normal(0.0,pow(self.innodes,-0.5),(self.hnodes,self.innodes))  
        self.who= numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)) 
        #self.activation_function= lambda x: return 1/(1-numpy.exp(-x))#lamdba--->匿名函数
        '''JOJO我不用匿名函数了'''

        '''scipy库有点问题，编译器报错，现在没解决想法'''
        '''哭了QAQ'''
        '''还是母鸡(资料少得可怜)'''
        '''或许要重新装一下scipy库'''
        '''但不想装'''
        '''大不了咱自己写（计划通）'''
        #pow( x, y )=>计算x的y次方
        #numpy.random.normal=>三个参数--->正态分布的中心，标准方差和行列式的行数和列数[作为一个元组传入
        pass
    def train(self,inputs_list,targets_list):
        inputs= numpy.array(inputs_list,ndmin= 2).T#1*3
        targets= numpy.array(targets_list,ndmin= 2).T#1*3
        hidden_input= numpy.dot(self.wih,inputs)#1*3
        hidden_outputs= self.sigmoid(hidden_input)#1*3
        finally_inputs= numpy.dot(self.who,hidden_outputs)#1*3
        finally_outputs= self.sigmoid(finally_inputs)#1*3
        output_error= targets_list- finally_outputs#1*3
        hidden_error= numpy.dot(self.who.T,output_error)
    #-----------------------------------------------------------------------------------------------
        self.who+= self.lr*numpy.dot((output_error*finally_outputs*(1.0-finally_outputs)),numpy.transpose(hidden_outputs)) 
        self.wih+= self.lr*numpy.dot((hidden_error*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs)) 
    #----------------------------------------------------------------------------------------------
    '''这就有点迷了'''
    def query(self,inputs_list):
        inputs= numpy.array(inputs_list,ndmin= 2).T#ndmin-->最小长度,'.T'--转置
                                                   #忘加.T了
        hidden_input= numpy.dot(self.wih,inputs) #矩阵相乘
        hidden_outputs= self.sigmoid(hidden_input)#Sigmoid抑制函数
        print("start")
        print(hidden_input)
        finally_inputs= numpy.dot(self.who,hidden_outputs)
        finally_outputs= self.sigmoid(finally_inputs)
        return finally_outputs
    def sigmoid(self,x):
        s= 1/(1-numpy.exp(-x))
        return s
def main():
    input("start")
    input_nodes= 3
    hidden_nodes= 3
    outout_nodes= 3
    learning_rate= 0.5
    n= neuralNetwork(input_nodes,hidden_nodes,outout_nodes,learning_rate)
    test001= n.query([1.0,-0.5,-1.5])

    pass
if __name__ == "__main__":
    input("start")
    main()
    
