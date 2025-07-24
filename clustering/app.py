from flask import Flask, render_template, request, redirect, session
from clustering import cluster as run_cluster
import secrets
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)


app.secret_key = os.getenv('SECRET_KEY')

@app.route("/")
def dashboard():
    cluster_count = session.get('cluster_count', 1)
    return render_template('dashboard.html', cluster_count=cluster_count)

@app.route('/select-risk', methods=['POST'])
def select_risk():
    risk_num = int(request.form['risk_num'])
    session['risk_num'] = risk_num
    #print(risk_num)

    cluster_count = session.get('cluster_count', 1)
    run_cluster(cluster_count,risk_num)

    return redirect('/')

@app.route('/cluster',methods=['POST'])
def handle_cluster():
    try:
        cluster_count = int(request.form['clusterCount'])
        session['cluster_count'] = cluster_count

        risk_num = session.get('risk_num')
        #print(risk_num)

        # if risk_num is None:
        #     risk_num = 1
        
        run_cluster(cluster_count,risk_num)
        #print("success")
    except Exception as e:
        print("Error generating clusters:", e)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)