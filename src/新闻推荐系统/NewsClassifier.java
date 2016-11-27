package 新闻推荐系统;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.converters.ConverterUtils.DataSource;


import java.io.FileInputStream; 
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.io.Serializable;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.wltea.analyzer.lucene.IKAnalyzer;

public class NewsClassifier implements Serializable {
	private static final long serialVersionUID = -384387983245497L;
	
	/*迄今收集训练数据*/
	private Instances m_Data = null;
	
	/*生成用于单词计数的过滤器*/
	private StringToWordVector m_Filter = new StringToWordVector();
	
	/*实际的分类器*/
	private NaiveBayesMultinomialUpdateable m_Classifier = new NaiveBayesMultinomialUpdateable();
	
	/*
	 加载数据并且构建一个初始化分类器
	 */
	public NewsClassifier() throws Exception{
		m_Data = DataSource.read("e:\\data.arff");
		
		m_Data.setClassIndex(m_Data.numAttributes()-1);
//		System.out.println(m_Data.classAttribute());
		
		//初始化过滤器
		m_Filter.setTFTransform(true);
		m_Filter.setIDFTransform(true);
		m_Filter.setInputFormat(m_Data);
		
		Instances filterdInstance = Filter.useFilter(m_Data, m_Filter);
		m_Classifier.buildClassifier(filterdInstance);
		System.out.println("初始分类器创建成功！");
	}
	
	/*
	 使用给定的训练模型更新模型
	 */
	public void updateData(String Content,String classValue) throws Exception{
		//把文本信息转换为实例
		Instance instance = makeInstance(Content,m_Data);
		
		//为实例设置类别值
		instance.setClassValue(classValue);
		
		m_Classifier.updateClassifier(instance);
		//将实例加入数据集
		m_Data.add(instance);
		
		//输出提示信息
		System.err.println("更新模型成功!");
	}
	
	/*
	 分类给定的文本信息
	 */
	public void classifyNews(String Content) throws Exception{
		//检查是否已构建分类器
		if(m_Data.numInstances() == 0)
		{
			throw new Exception("没有分类器可用!");
		}
		
		//检查是否分类器和过滤器为最新
//		if(!m_UpToDate){
//			//初始化过滤器，并告知输入格式
//			m_Filter.setInputFormat(m_Data);
//			
//			//从训练数据生成单词计数
//			Instances filteredData = Filter.useFilter(m_Data, m_Filter);
//			
//			//重建分类器
//			m_Classifier.buildClassifier(filteredData);
//			m_UpToDate = true;
//		}
		
		//形成单独的小测试集,所以该文本信息不会添加到m_Data的字符串属性中
		Instances testset = m_Data.stringFreeStructure();
		
		//使文本信息成为测试实例
		Instance instance = makeInstance(Content,testset);
		
		//过滤实例
		m_Filter.input(instance);
		Instance filterdInstance = m_Filter.output();
		
		
		//获取预测类别值的索引
		double predicted = m_Classifier.classifyInstance(filterdInstance);
		
		//输出类别值
		System.err.println("文本信息分类为:"+m_Data.classAttribute().value((int)predicted));
		
	}
	
	/*
	 将文本信息转换为实例的方法
	 */
	private Instance makeInstance(String Content,Instances data){
		//创建一个属性数量为2，权重为1，全部值都为缺失的实例
		Instance instance = new DenseInstance(2);
		
		//设置文本信息属性的值
		Attribute contentAtt = data.attribute(m_Data.attribute(0).name());
		instance.setValue(contentAtt, contentAtt.addStringValue(Content));
		
		//让实例能够访问数据集中的属性信息
		instance.setDataset(data);
		return instance;
	}
	
	public static void main(String[] options){
		try{
			
			//读入文本信息文件，存储为字符串
			String newsName = Utils.getOption('m',options);
			if(newsName.length() == 0){
				throw new Exception("必须提供文本信息文件的名称");
			}
			FileReader m = new FileReader(newsName);
			StringBuffer content = new StringBuffer();
			int l;
			while((l = m.read())!= -1){
				content.append((char)l);
			}
			m.close();
			
			//检查是否文本为英文
			boolean isEnglish = Utils.getFlag('E', options);
			if(!isEnglish){
				//只有汉字需要进行中文分词
			   Analyzer ikAnalyzer = new IKAnalyzer();
			   Reader reader = new StringReader(content.toString());
			   TokenStream stream = (TokenStream) ikAnalyzer.tokenStream("", reader);
			   CharTermAttribute termAtt = (CharTermAttribute)stream.addAttribute(CharTermAttribute.class);
			   content = new StringBuffer();
			   while(stream.incrementToken()){
				   content.append(termAtt.toString() + " ");//当初败在这里，少加了个空格，让你浪费了多少时间！
			   }
			}
			
			//检查是否已给定类别值
			String classValue = Utils.getOption('C', options);
			//如果模型文件存在，则读入，否则创建新的模型文件
			String modelName = Utils.getOption('t', options);
			if(modelName.length() == 0){
				throw new Exception("必须提供模型文件的名称");
			}
			NewsClassifier newsCl;
			try{
				ObjectInputStream modelInObjectFile = new ObjectInputStream(new FileInputStream(modelName));
				newsCl = (NewsClassifier) modelInObjectFile.readObject();
				modelInObjectFile.close();
			}catch(FileNotFoundException e){
				newsCl = new NewsClassifier();
			}
			
			//处理文本信息
			if(classValue.length()!= 0){
				newsCl.updateData(content.toString(), classValue);
			}else{
				newsCl.classifyNews(content.toString());
			}
			
			//保存文本信息分类器对象
			ObjectOutputStream modelOutobjectFile = new ObjectOutputStream(new FileOutputStream(modelName));
			modelOutobjectFile.writeObject(newsCl);
			modelOutobjectFile.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
