from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline
with open("D:\\internships\\cellula\Hotel prediction\\model.pkl", 'rb') as model_file:
    pipeline = pickle.load(model_file)


def is_valid_integer(value):
    return str(value).isdigit()


@app.route('/')
def home():
    return render_template('index.html')
 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and validate as integers
        number_of_week_nights = request.form['number of week nights']
        lead_time = request.form['lead time']
        average_price = request.form['average price']
        special_requests = request.form['special requests']
        day_of_reservation = request.form['day of reservation']
        month_of_reservation = request.form['month of reservation']

        if all(is_valid_integer(val) for val in [number_of_week_nights, lead_time, average_price,
                                                 special_requests, day_of_reservation, month_of_reservation]):
            # Convert validated inputs to integers
            number_of_week_nights = int(number_of_week_nights)
            lead_time = int(lead_time)
            average_price = int(average_price)
            special_requests = int(special_requests)
            day_of_reservation = int(day_of_reservation)
            month_of_reservation = int(month_of_reservation)

            # Create a DataFrame from user-entered features
            user_features = pd.DataFrame({
                'number of week nights': [number_of_week_nights],
                'lead time': [lead_time],
                'average price': [average_price],
                'special requests': [special_requests],
                'day of reservation': [day_of_reservation],
                'month of reservation': [month_of_reservation]
            })

            # Use the trained pipeline to make predictions
            prediction = pipeline.predict(user_features)

            # Assuming 1 indicates canceled and 0 indicates not canceled in your prediction
            def map_to_status(prediction):
                if prediction == 1:
                    return "Canceled"
                else:
                    return "Not Canceled"
                
            # Render the template with the prediction value
        
            predicted_status = map_to_status(prediction[0])
            return render_template('index.html', prediction=f'Predicted Status: {predicted_status}')

        else:
            raise ValueError("An error occurred: Please enter valid integer numbers.")

    except ValueError as e:
        error_message = str(e)
        return render_template('index.html', error_message=error_message)
    

if __name__ == '__main__':
    app.run(debug=True)
