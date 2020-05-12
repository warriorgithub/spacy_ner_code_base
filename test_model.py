from flask import Flask,request
import spacy
import pdftotextimport json
import logging

app = Flask(__name__)

@app.route('/fetchskills', methods=['GET'])
def extractSkills():# Test the trained model
	resumeText = 'Hello this is my java resume, I am good with NLTK, nlp and machine learning. I worked with Php,Sql, data science, numpy, pandas,spring,airflow.'
	output_dir = 'model'  
		#Where you have saved your train model
	print("Loading from", output_dir)
	nlp2 = spacy.load(output_dir)
	doc2 = nlp2(resumeText)
	#print("doc  ",doc2.ents)
	response = {}
	text=""
	for ent in doc2.ents:
		text+= ent.texttext+=" "
		response['Skills'] = text
		return response


if __name__ == '__main__':
	app.run(debug=True)