from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

def transform(intl_features):
    '''Transforms the input features into a form which is understanded by our model.
       The html file takes the name of the city, brand whether the person is the first, second, third or fourth or more owner
       as a string. So we need to encode these strings so that they can be feeded into our model.'''
    l = []
    cities = ['24 Pargana', 'Abohar', 'Adalaj', 'Adoni', 'Adyar', 'Agra',
       'Ahmedabad', 'Ahmednagar', 'Ajmer', 'Akot', 'Alappuzha', 'Alibag',
       'Aligarh', 'Alipore', 'Allahabad', 'Aluva', 'Alwar', 'Ambala',
       'Ambikapur', 'Amraoti', 'Amravati', 'Amritsar', 'Anand',
       'Anantapur', 'Anantnag', 'Anekal', 'Anjar', 'Ankleshwar', 'Aquem',
       'Arkalgud', 'Arrah', 'Asansol', 'Aurangabad', 'Azamgarh',
       'Badarpur', 'Badaun', 'Bagalkot', 'Baghpat', 'Bahadurgarh',
       'Bahadurpur', 'Balaghat', 'Bally', 'Baloda', 'Balrampur',
       'Bangalore', 'Banka', 'Banki', 'Bankura', 'Barabanki', 'Baran',
       'Barasat', 'Bardhaman', 'Bareilly', 'Bargarh', 'Baripara', 'Basti',
       'Batala', 'Begusarai', 'Belgaum', 'Bellary', 'Berhampore',
       'Berhampur', 'Betul', 'Bharatpur', 'Bharuch', 'Bhatinda',
       'Bhavnagar', 'Bhawani Mandi', 'Bhilai Nagar', 'Bhilwara',
       'Bhiwadi', 'Bhiwandi', 'Bhiwani', 'Bhopal', 'Bhubaneshwar',
       'Bhubaneswar', 'Bhuj', 'Bidar', 'Bihar Shariff', 'Bijapur',
       'Bijnor', 'Bikaner', 'Bilaspur', 'Bodhan', 'Bokaro', 'Bolpur',
       'Budhlada', 'Bulandshahr', 'Bundi', 'Burdwan', 'Buxar', 'Calicut',
       'Cannanore (kannur)', 'Chakan', 'Chaksu', 'Challakere',
       'Chandigarh', 'Chandrapur', 'Chatrapur', 'Chenani', 'Chennai',
       'Chhatarpur', 'Chhindwara', 'Chikamaglur', 'Chikkaballapur',
       'Chinchwad', 'Chinsurah', 'Chitradurga', 'Churu', 'Coimbatore',
       'Cuttack', 'Dadra & Nagar Haveli', 'Dadri', 'Dakshina Kannada',
       'Darbhanga', 'Davanagere', 'Deesa', 'Dehradun', 'Delhi', 'Deoghar',
       'Deolali', 'Deoria', 'Dhamtari', 'Dhanbad', 'Dharamasala',
       'Dhariawad', 'Dharmapuri', 'Dharmavaram', 'Dharwad', 'Dharwar',
       'Dhubri', 'Dibrugarh', 'Dongargaon', 'Dungarpur', 'Durg',
       'Durgapur', 'Dwarka', 'Ernakulam', 'Erode', 'Falakata',
       'Faridabad', 'Faridkot', 'Farrukhabad', 'Farukhabad', 'Ferozepur',
       'Gadarpur', 'Gadchiroli', 'Gadwal', 'Ganaur', 'Gandhidham',
       'Gandhinagar', 'Gangaghat', 'Gangaikondan', 'Ganganagar',
       'Gangtok', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur',
       'Goa-panaji', 'Godavari', 'Godhara', 'Gohana', 'Gondia',
       'Gorakhpur', 'Goregaon', 'Guntur', 'Gurdaspur', 'Gurgaon',
       'Guwahati', 'Gwalior', 'Haldwani', 'Hamirpur', 'Hamirpur(hp)',
       'Hanumangarh', 'Haridwar', 'Herbertpur', 'Hisar', 'Hissar',
       'Honavar', 'Hooghly', 'Hoshiarpur', 'Hospet', 'Hosur', 'Howrah',
       'Hubli', 'Hyderabad', 'Idukki', 'Indi', 'Indore', 'Jabalpur',
       'Jagdalpur', 'Jaipur', 'Jaisalmer', 'Jajpur', 'Jalandhar',
       'Jalaun', 'Jalgaon', 'Jamalpur', 'Jammu', 'Jamnagar', 'Jamshedpur',
       'Jamtara', 'Jatani', 'Jaunpur', 'Jhajjar', 'Jhalawar', 'Jhansi',
       'Jhumri Tilaiya', 'Jhunjhunu', 'Jind', 'Jobner', 'Jodhpur',
       'Jorhat', 'Junagadh', 'Kachchh', 'Kadapa', 'Kadi', 'Kaithal',
       'Kalyan', 'Kanchipuram', 'Kanpur', 'Kanpur Nagar', 'Kanyakumari',
       'Karim Nagar', 'Karnal', 'Kartarpur', 'Karwar', 'Kasargode',
       'Kasba', 'Kathua', 'Katihar', 'Katni', 'Kendua', 'Khalilabad',
       'Khandela', 'Khandwa', 'Kharagpur', 'Kharar', 'Kheda',
       'Khedbrahma', 'Kochi', 'Kolar', 'Kolhapur', 'Kolkata', 'Kollam',
       'Koppal', 'Kota', 'Kotdwar', 'Kottayam', 'Krishna', 'Krishnagar',
       'Kullu', 'Kurnool', 'Kurukshetra', 'Lansdowne', 'Latur',
       'Lonavala', 'Lucknow', 'Ludhiana', 'Madurai', 'Malout', 'Manali',
       'Mandi', 'Mandi Dabwali', 'Mandya', 'Mangalore', 'Mansa',
       'Marandahalli', 'Margao', 'Mathura', 'Medak', 'Meerut', 'Mehsana',
       'Mettur', 'Mohali', 'Mohammadabad', 'Moradabad', 'Morbi',
       'Motihari', 'Mubarakpur', 'Mughalsarai', 'Muktsar', 'Mumbai',
       'Murad Nagar', 'Muvattupuzha', 'Muzaffarnagar', 'Muzaffarpur',
       'Mysore', 'Nabha', 'Nadiad', 'Nagaon', 'Nagaur', 'Nagpur',
       'Naihati', 'Nalagarh', 'Namakkal', 'Nanded', 'Nanjangud',
       'Naraingarh', 'Narnaul', 'Nashik', 'Navi Mumbai', 'Navsari',
       'Nawanshahr', 'Nayagarh', 'Nazira', 'Nellore', 'Nizamabad',
       'Noida', 'Osmanabad', 'Palai', 'Palakkad', 'Palamu', 'Palanpur',
       'Pali', 'Palwal', 'Panchkula', 'Panipat', 'Panvel', 'Parola',
       'Pathankot', 'Patiala', 'Patna', 'Perumbavoor', 'Phagwara',
       'Pinjore', 'Pondicherry', 'Poonamallee', 'Porbandar', 'Pratapgarh',
       'Pune', 'Puri', 'Purnia', 'Puttur', 'Qadian', 'Raiganj', 'Raigarh',
       'Raigarh(mh)', 'Raipur', 'Raiwala', 'Rajahmundry', 'Rajkot',
       'Rajnandgaon', 'Rajouri', 'Ramanagar', 'Ranchi', 'Ranga Reddy',
       'Rangpo', 'Ranip', 'Ranoli', 'Rasra', 'Ratnagiri', 'Rewari',
       'Rohtak', 'Roorkee', 'Rudrapur', 'Rupnagar', 'Saharanpur', 'Salem',
       'Samastipur', 'Sambalpur', 'Sanand', 'Sangareddy', 'Sangli',
       'Sangrur', 'Sant Kabir Nagar', 'Satara', 'Satna', 'Secunderabad',
       'Seppa', 'Sheikhpura', 'Shillong', 'Shimla', 'Shimoga', 'Shivpuri',
       'Sholapur', 'Sibsagar', 'Sidhi', 'Silchar', 'Siliguri', 'Silvasa',
       'Simdega', 'Singhbhum', 'Sirsa', 'Sirsi', 'Sitapur', 'Siwan',
       'Solan', 'Solapur', 'Sonepat', 'Sonipat', 'Sri Ganganagar',
       'Srinagar', 'Sultanpur', 'Sundargarh', 'Surat', 'Surendranagar',
       'Suri', 'Swaimadhopur', 'Thane', 'Thangadh', 'Thanjavur',
       'Thiruvallur', 'Thiruvananthapuram', 'Thrissur', 'Tikamgarh',
       'Tiruchirappalli', 'Tirunelveli', 'Tiruppur', 'Tiruvallur',
       'Tiruverkadu', 'Trivandrum', 'Tumkur', 'Udaipur', 'Udaipurwati',
       'Udhampur', 'Udupi', 'Ujjain', 'Uluberia', 'Una', 'Unnao',
       'Uppidamangalam', 'Uran', 'Vadodara', 'Valsad', 'Vandalur', 'Vapi',
       'Varanasi', 'Vasai', 'Vastral', 'Vellore', 'Vidisha', 'Vijayawada',
       'Viramgam', 'Virar', 'Virudhunagar', 'Visakhapatnam',
       'Vizianagaram', 'Warangal', 'Wardha', 'Yamuna Nagar', 'Yemmiganur',
       'Zirakpur']
    brands = ['BMW', 'Bajaj', 'Benelli', 'Ducati', 'Harley-Davidson', 'Hero',
       'Honda', 'Hyosung', 'Ideal', 'Indian', 'Jawa', 'KTM', 'Kawasaki',
       'LML', 'MV', 'Mahindra', 'Rajdoot', 'Royal Enfield', 'Suzuki',
       'TVS', 'Triumph', 'Yamaha', 'Yezdi']
    x = [0 for _ in range(23)]
    a = [0 for _ in range(443)]
    for i in range(6):
        if i == 3:
            a[cities.index(intl_features[i])] = 1
            l.extend(a)
        elif i == 4:
            if intl_features[i] == "FIRST OWNER":
                x.extend([1,0,0,0])
            elif intl_features[i] == "SECOND OWNER":
                x.extend([0,0,1,0])
            elif intl_features[i] == "THIRD OWNER":
                x.extend([0,0,0,1])
            else:
                x.extend([0,1,0,0])
        elif i == 5:
            x[brands.index(intl_features[i])] = 1
            l.extend(x)
        else:
            l.append(float(intl_features[i]))
    final_features = [np.array(l)]
    return final_features

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods = ['POST'])
def predict():
    intl_features = [x for x in request.form.values()]
    final_features = transform(intl_features)
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('index.html',prediction_text = "Price of bike is :{}".format(output))

if __name__ == "__main__":
    app.run(debug=True)