#include<iostream>
#include<vector>
#include<queue>
#include<cmath> 
#include<cstdlib>
#include<ctime>
#define DEBUG
#define isTrace 0
using namespace std;
//****************************basic****************************************
#if isTrace==1
#define Trace(m) cout<<#m"="<<(m)<<endl;
#else
#define Trace(m) 
#endif
#ifdef DEBUG
#define ASSERT(cond) if (!(cond)) {cout<<"ASSERT FAIL: "#cond"="<<(cond)<<endl;system("pause");}
#define ASSERT_M(cond,mess) if (!(cond)) {cout<<"ASSERT FAIL: "#cond"="<<(cond)<<" Mess:"<<mess<<endl;system("pause");}

#else
#define ASSERT(cond)
#define ASSERT_M(cond,mess)
#endif
namespace ActivateFun{
static inline double sign(double v);
static inline double sign(double v){
	//-1~1
	return v>0?1:-1;
}

//sigmoid
static inline double logistic(double v){
	//0~1 signmoid
	static const int a=1;
	return 1.0/(1.0+exp(-a*v));
}
static inline double tanh(double v){
	//-1~1 signmoid
	return (1.0-exp(-2.0*v))/(1.0+exp(-2.0*v));
}
}
struct Cell_Global{
	int time;
	typedef double (*Fun)(double);
	Fun activeFun;
	const double initLearnEff;
public:
	Cell_Global():time(0),activeFun(NULL),initLearnEff(0.3){
		activeFun=ActivateFun::sign;
	}
	double getLearnEff()const{
		return initLearnEff/(double)(time+1);
	}
	static bool reachThrehold(double v){
		return v>0;}
	
public:
	void update(){time++;}
	double p(double t){if(activeFun==NULL)return t;return activeFun(t);}
};
class Cell{
#ifdef DEBUG
#define Dump(m) cout<<m;
#else 
//#define Dump(m)
#endif
public:
	static Cell_Global *global;
	Cell():time(0),index(0){}
	//need to override.
public:

#ifdef Dump(m) 
	virtual void dump()=0;
#endif
	
	virtual void update(){time=global->time;}
	virtual double getValue()const=0;
	virtual void setValue(double v)=0;
	virtual void outputTo(Cell*c)=0;
	virtual void receiveFrom(Cell*c)=0;
	virtual bool isReady()const{
		return time==global->time;
	}
	virtual int getIndex(){return index;}
	virtual void setIndex(int t){index=t;}
	//
	//***********
public:

	void connectTo( Cell *c){
		outputTo(c);
		c->receiveFrom(this);
	}
	int getTime()const{return time;}
protected:
	void setTime(int t){time=t;}
	void resetTime(){time=0;}
private:
	int time;
	int index;
};

typedef vector<Cell *> Cells;
Cell_Global *Cell::global=new Cell_Global();
//
class InvariantCell:public Cell{
	Cells outs;
	double invariantValue;
public:
#ifdef Dump(m)
	virtual void dump(){
		Dump("InvariantCell_"<<getIndex()<<"[");
		int i;
		Dump("value:");Dump(this->getValue());
		Dump(" ");
		Dump("to:");
		for(i=0;i<(int)outs.size();i++){
			Dump(outs[i]->getIndex()<<",");
		}
		Dump("]");
	}
#endif
	InvariantCell(double v=1):invariantValue(v){}
	virtual void update(){}
	virtual double getValue()const{
		return invariantValue;
	}
	virtual void setValue(double v){invariantValue=v;}
	virtual void outputTo(Cell*c){
		outs.push_back(c);
	}
	virtual bool isReady()const{
		return true;
	}
private:
	virtual void receiveFrom(Cell*c){
		ASSERT_M(false,"InvariantCell can't invoke receiveFrom(Cell*)");
	}
};
class InputCell:public Cell{
	Cells outs;
	double value;
public:
#ifdef Dump(m)
	virtual void dump(){
		Dump("InputCell_"<<getIndex()<<"[");
		int i;
		Dump("value:");Dump(this->getValue());
		Dump(" ");
		Dump("to:");
		for(i=0;i<(int)outs.size();i++){
			Dump(outs[i]->getIndex()<<",");
		}
		Dump("]");
	}
#endif
	InputCell(double v=1.0):value(v){}
	virtual void update(){
		Cell::update();
		//propagation to next neuralCell.
		for(int i=0;i<(int)outs.size();i++){
			outs[i]->update();
		}}
	virtual double getValue()const{
		return value;
	}

	virtual void setValue(double v){value=v;}
	void inputValue(double v){value=v;}
	virtual void outputTo(Cell*c){
		outs.push_back(c);
	}
private:
	virtual void receiveFrom(Cell*c){
		ASSERT_M(false,"InputCell can't invoke receiveFrom(Cell*)");
	}
};
class NeuralCell:public Cell{
private:
	Cells outs;
	Cells ins;
	vector<double>inWeights;
	double value;
	Cell_Global::Fun p;
	//
	double bais;
public:
	#ifdef Dump(m)
	virtual void dump(){
		Dump("NeuralCell_"<<getIndex()<<"[");
		int i;
		Dump("from:");
		for(i=0;i<(int)ins.size();i++){
			Dump(ins[i]->getIndex()<<"/"<<inWeights[i]<<",");
		}
		Dump(" ");
		Dump("to:");
		for(i=0;i<(int)outs.size();i++){
			Dump(outs[i]->getIndex()<<",");
		}
		Dump("]");
	}
#endif
	Cells& getIns(){return ins;}
	virtual void outputTo(Cell*c){
		outs.push_back(c);
	}
	virtual void receiveFrom(Cell*c){
		ins.push_back(c);
		inWeights.push_back(0);
	}
public:
	vector<double>&getInWeights(){return inWeights;}
	virtual double getValue()const{return value;}
	virtual void setValue(double v){value=v;}
	NeuralCell():p(NULL),bais(0),value(0){}

	virtual void update(){
		int i=0;
		if(!ins.empty()){//if not empty,update the value.
			for(i=0;i<(int)ins.size();i++){//if not all ins are ready,wait.
				if(!ins[i]->isReady())return;
			}
			//all is ready.
			//update this cell
			value=0;
			for(i=0;i<(int)ins.size();i++){
				value+=ins[i]->getValue()*inWeights[i];
			}
			if(p==NULL)value=global->p(value);else value=p(value);
		}
		Cell::update();
		//propagation to next neuralCell.
		for(i=0;i<(int)outs.size();i++){
			outs[i]->update();
		}
	}
};

class OutputCell:public Cell{
Cell*in;
	double value;
public:
#ifdef Dump(m)
	virtual void dump(){
		Dump("OutputCell_"<<getIndex()<<"[");
		Dump("value:");Dump(this->getValue());
		Dump(" ");
		Dump("from:");
			Dump(in->getIndex()<<",");
		Dump("]");
	}
#endif
	OutputCell(double v=0):value(v),in(NULL){}
	virtual void update(){
		value=in->getValue();
		Cell::update();
	}
	virtual double getValue()const{
		return value;
	}
	virtual void setValue(double v){value=v;}
	virtual void receiveFrom(Cell*c){
		in=c;
	}
private:
	virtual void outputTo(Cell*c){
		ASSERT_M(false,"OutputCell can't invoke outputTo(Cell*)");
	}
};

class DelayCell:public Cell{
private:
	int delayN;
	Cell* in;
	Cells outs;
	struct ValueHistory{
		double value;
		int time;
	};
	queue<ValueHistory>buffer;
	virtual void setValue(double v){ASSERT(false);}
public:
	#ifdef Dump(m)
	virtual void dump(){
		Dump("DelayCell_"<<getIndex()<<"[");
		int i;
		Dump("from:");in->dump();
		Dump(" ");
		Dump("to:");
		for(i=0;i<(int)outs.size();i++){
			Dump(outs[i]->getIndex()<<",");
		}
		Dump("]");
	}
#endif
	virtual double getValue(){
		ASSERT((global->time-buffer.front().time)==this->delayN);
		return buffer.front().value;
	}
	DelayCell(int delay=1):delayN(delay){
		for(int i=0;i<(int)delayN;i++){
			ValueHistory vh;
			vh.time=i-delayN+1;
			vh.value=0;
			buffer.push(vh);
			
		}
	}
	virtual void update(){
		ValueHistory vh;
		vh.time=in->getTime();
		vh.value=in->getValue();
		buffer.push(vh);buffer.pop();
		Cell::update();
	}
	virtual bool isReady()const{
		return (global->time-getTime())<=delayN;
	}

};


template<typename CellType=NeuralCell>
class Layer:public Cell{
private:
	int n;//numCells;
	typedef vector<CellType*> LayerCells;
	LayerCells cells;
	//
	InvariantCell constCell;//provide the +1,and let w0 be bias.
public:
	//need to override.
	Layer():n(0){}
	Layer(int tn):n(0){addCells(tn);}
	int getN()const{return n;}
	void setN(int tn){n=tn;}
public:
	#ifdef Dump(m)
	virtual void dump(){
		Dump("Layer_"<<getIndex()<<"{");
		for(int i=0;i<(int)getCells().size();i++){
			getCells()[i]->dump();Dump(" ");
			
		}Dump("}");
	}
#endif
	virtual bool isReady()const{
		for(int i=0;i<(int)cells.size();i++){
			if(!cells[i]->isReady())return false;
		}
		return true;
	}
	virtual void update(){
		int i;
		for(i=0;i<(int)cells.size();i++)cells[i]->update();
		Cell::update();
	}
	LayerCells&getCells(){return cells;}
	virtual void addCells(int num){
		while(num>0){
			CellType*cell=new CellType();
			constCell.connectTo(cell);
			getCells().push_back(cell);
			getCells().back()->setIndex(n);
			n++;
			num--;
		}
	}
private:
	virtual double getValue()const{ASSERT(false);return 0;}
	virtual void setValue(double v){ASSERT(false);}
public:
	virtual void outputTo(Cell*c){
		int i;
		for(i=0;i<(int)cells.size();i++){
			c->receiveFrom(cells[i]);
		}
	}
	virtual void receiveFrom(Cell*c){
		int i;
		for(i=0;i<(int)cells.size();i++){
			c->outputTo(cells[i]);
		}
	}

};
typedef vector<double> Input;
typedef vector<double> Output;

class InputLayer:public Layer<InputCell>{
public:
	InputLayer(int n):Layer(0){addCells(n);}
	void inputData(const Input&values){
		ASSERT(this->getN()==values.size());
		for(int i=0;i<(int)getN();i++)this->getCells()[i]->inputValue(values[i]);
	}
	#ifdef Dump(m)
	void dump(){
		Dump("Input");Layer::dump();
	}
#endif
	virtual void addCells(int num){
		while(num>0){
			InputCell*cell=new InputCell();
			getCells().push_back(cell);
			getCells().back()->setIndex(this->getN());
			this->setN(this->getN()+1);
			num--;
		}
	}
};
class OutputLayer:public Layer<OutputCell>{
public:
	OutputLayer():Layer(0){}
	Output outputData(){
		Output output;
		for(int i=0;i<(int)getN();i++){
			output.push_back(getCells()[i]->getValue());
		}
		return output;
	}
	#ifdef Dump(m)
	void dump(){
		Dump("Output");Layer::dump();
	}
#endif
	public:
		virtual void receiveFrom(Cell*c){
		if(dynamic_cast<NeuralCell*>(c)){
			this->addCells(1);
			c->connectTo(this->getCells().back());
		}
	}
private:
	virtual void outputTo(Cell*c){
		ASSERT(false);
	}
};

//*****************************end basic*************************************

//neural network.
struct TrainPair{
	Input ins;
	Output outs;
	void clear(){ins.clear();outs.clear();}
};
typedef vector<TrainPair> TrainPairs;
class NN:public Cell{
	InputLayer inputLayer;
	vector<Layer<NeuralCell>* >hiddenLayers;
	OutputLayer outputLayer;
	int numLayers;
	int numIns;
	int numOuts;
public:
	#ifdef Dump(m)
	void dump(){
		inputLayer.dump();
		Dump("\n");
		/*outputLayer.dump();*/
		hiddenLayers.back()->dump();
		Dump("\n");
	}
#endif
public:
	NN(int n,int m):inputLayer(n),numIns(n),numOuts(m){construct();}
	void construct(){
		hiddenLayers.push_back(new Layer<NeuralCell>(1));
		inputLayer.connectTo(hiddenLayers[0]);
		hiddenLayers.back()->connectTo(&outputLayer);
	}
	//
	void inputData(Input inputs){ASSERT(inputs.size()==inputLayer.getN());inputLayer.inputData(inputs);}
	void update(){
		Cell::global->update();
		inputLayer.update();
		ASSERT(inputLayer.isReady());
		ASSERT(outputLayer.isReady());
	}
	Output outputData(){
		ASSERT(outputLayer.isReady());
		return outputLayer.outputData();
	}
public:
	void trains(const TrainPairs& pairs){
		for(int k=0;k<(int)pairs.size();k++){
			train(pairs[k]);
		}
	}
	void train(const TrainPair& pair){
		ASSERT(pair.ins.size()==inputLayer.getN());
		ASSERT(pair.outs.size()==outputLayer.getN());
		double learnEff=global->getLearnEff();
		Trace(learnEff);
		this->inputData(pair.ins);
		this->update();
		Output outputs=this->outputData();
		for(int i=0;i<(int)outputs.size();i++){
			//foreach ouput,refresh weight.
			double e=pair.outs[i]-outputs[i];//desire-evaluate.
			//NeuralCell*&cell=outputLayer.getCells()[i];
			NeuralCell*&cell=hiddenLayers.back()->getCells()[i];
			vector<double>&weights=cell->getInWeights();
			for(int j=0;j<(int)weights.size();j++){
				Trace(e);
				Trace(cell->getIns()[j]->getValue());
				Trace(cell->getInWeights()[j]);
				cell->getInWeights()[j]+=learnEff*e*cell->getIns()[j]->getValue();
			}
		}
	}
		//not used.
private:
	virtual double getValue()const{return 0;}
	virtual void setValue(double v){}
	virtual void outputTo(Cell*c){}
	virtual void receiveFrom(Cell*c){}
};


void printVector(const vector<double>&as,char split=','){
	for(int i=0;i<(int)as.size();i++){cout<<as[i];if(i!=as.size()-1)cout<<split;}
}

void analyseLine(char *buffer,vector<double>&nums,int numNums=0){
	int t=0;
	for(int i=0;i<(int)strlen(buffer);i++){
		if(buffer[i]!=' '){
			if(sscanf(buffer+i,"%d",&t)==1)
				nums.push_back(t);
			i++;
			while(buffer[i]&&buffer[i]!=' ')i++;
		}else{
		}
	}
	
}
void printPair(const TrainPair&pair){
	cout<<"pair{ins:[";printVector(pair.ins,' ');cout<<"],\touts:[";printVector(pair.outs,' ');cout<<"]};";
}
void readInput(TrainPair&pair,int numIn=-1){
	pair.ins.clear();
	char buffer[100];
	cout<<"In(\\n end):";
	if(numIn==-1){
		cin.getline(buffer,100,'\n');
		analyseLine(buffer,pair.ins);
	}else{
		for(int i=0;i<(int)numIn;i++){
			int temp;
			cin>>temp;
			pair.ins.push_back(temp);
		}
	}
}
void readOutput(TrainPair&pair,int numOut=-1){
	pair.outs.clear();
	char buffer[100];
	cout<<"Output(\\n end):";
	if(numOut==-1){
	cin.getline(buffer,100,'\n');
	analyseLine(buffer,pair.outs);
	}else{
		for(int i=0;i<(int)numOut;i++){
			int temp;
			cin>>temp;
			pair.outs.push_back(temp);
		}
	}
}
void readPair(TrainPair&pair,int numIn=-1,int numOut=-1){
	readInput(pair,numIn);
	readOutput(pair,numOut);
}
/*
2 5
1
8 9
1 
-7 -5
-1
-5 -4
-1
12 13
1
12 13
2 5
-7 -5



*/
int main(){
	const int NumIn=2;
	const int NumOut=1;
	NN nn=NN(NumIn,NumOut);
	nn.dump();
	/*TrainPairs pairs;
	nn.trains(pairs);*/

	TrainPair pair;
	int count=0;
	int numA;int numB;
	//cout<<"Num training cases:";cin>>t;cin.get();
	TrainPairs trainSets;
	pair.outs.push_back(1);

	cout<<"num of Set A:";cin>>count;
	numA=count;
	pair.outs[0]=1;
	while(count--){readInput(pair,NumIn);trainSets.push_back(pair);}
	cout<<"num of Set B:";cin>>count;
	numB=count;
	pair.outs[0]=-1;
	while(count--){readInput(pair,NumIn);trainSets.push_back(pair);}
	srand((unsigned int)time(NULL));
	for(int i=0;i<(int)trainSets.size()*100;i++){
		nn.train(trainSets[rand()%trainSets.size()]);
	}
	//nn.trains(trainSets);
	cout<<"Training samples{"<<endl;
	cout<<numA<<endl;
	for(int i=0;i<(int)numA;i++){
		printVector(trainSets[i].ins,' ');cout<<endl;
	}
	cout<<numB<<endl;
	for(int i=numA;i<(int)numA+numB;i++){
		printVector(trainSets[i].ins,' ');cout<<endl;
	}
	cout<<"}"<<endl;
	nn.dump();
	//
	count=0;
	cout<<"**********Verifying**********..."<<endl;
	vector<TrainPair>testCases;
	cout<<"0 0 means prints testCases"<<endl;
	while(1){
		pair.clear();
		if(count>100)break;
		//cout<<"Case "<<count<<":"<<endl;
		readInput(pair,NumIn);
		nn.inputData(pair.ins);
		if(pair.ins[0]==0&&pair.ins[1]==0){
			cout<<"testCases:{"<<endl;
			int i;
			for( i=0;i<(int)testCases.size();i++){
				if(testCases[i].outs[0]==1){printVector(testCases[i].ins,' ');cout<<"\t\t belong to "<<(Cell_Global::reachThrehold(testCases[i].outs[0])?"A":"B");cout<<endl;
				}}for( i=0;i<(int)testCases.size();i++){
				if(!(testCases[i].outs[0]==1)){printVector(testCases[i].ins,' ');cout<<"\t\t belong to "<<(Cell_Global::reachThrehold(testCases[i].outs[0])?"A":"B");cout<<endl;
				}}
			cout<<"}"<<endl;
			continue;
		}
		nn.update();
		pair.outs=nn.outputData();
		cout<<"[";printVector(pair.ins,' ');cout<<"] belong to "<<(Cell_Global::reachThrehold(pair.outs[0])?"A":"B");
		testCases.push_back(pair);
		//printPair(pair);
		cout<<endl;
		count++;
	}
	//


	//

	return 0;
}