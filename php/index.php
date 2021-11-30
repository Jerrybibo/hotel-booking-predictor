<!DOCTYPE html>
<html lang="en-US">
<head>
    <title>CS 334 - Machine Learning</title>
    <!-- meta tags -->
    <meta charset="utf=8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Include jQuery -->
    <script type="text/javascript" src="https://code.jquery.com/jquery-1.11.3.min.js"></script>
    <!-- Bootstrap CSS + JS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
    <!-- Bootstrap date picker -->
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-datepicker/1.4.1/js/bootstrap-datepicker.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-datepicker/1.4.1/css/bootstrap-datepicker3.css"/>
    <!-- Date Picker JS -->
    <script>
        $(document).ready(function(){
            const date_input = $('input[name="arrival_date"]');
            const container = "body";
            const options = {
                format: 'mm/dd',
                container: container,
                autoclose: true,
                orientation: 'top'
            };
            date_input.datepicker(options);
        })
    </script>

</head>
<body>
    <div class="container m-4">
        <h2>Hotel Booking Dataset
            <small class="text-muted">Interactive UI</small>
        </h2>
        <ul>
            <li>Several features are excluded from this form due to weak predictive power, or very rare deviations from average/majority value.</li>
            <li>Hover over the feature title text to learn more.</li>
        </ul>
        <form action="" method="post">
            <table class="table table-bordered" style="text-align:center">
                <thead>
                    <tr>
                        <th data-bs-toggle="tooltip" title="Number of days that elapsed between the entering date of the booking into the management system and the arrival date.">Lead Time</th>
                        <th data-bs-toggle="tooltip" title="Month and year of arrival date.">Arrival Date</th>
                        <th data-bs-toggle="tooltip" title="Number of days that elapsed between arrival date and checkout date.">Stay Duration</th>
                        <th data-bs-toggle="tooltip" title="Number of adults.">Adults</th>
                        <th data-bs-toggle="tooltip" title="Number of minors. (<18)">Minors</th>
                        <th data-bs-toggle="tooltip" title="Whether the traveler is from abroad. Note that this data set uses hotel information from Portugal, so abroad is equivalent to not from Portugal.">From Abroad?</th>
                        <th data-bs-toggle="tooltip" title="Type of hotel.">Hotel Type</th>
                        <th data-bs-toggle="tooltip" title="Type of deposit, if any.">Deposit Type</th>
                        <th data-bs-toggle="tooltip" title="Type of customer. Transient: Singular booking, not part of group or contract; Transient-Party: Multiple transient booking (rooms), but not part of contract; Group: Associated with a set group; Contract: Contract-bound">Customer Type</th>

                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><input type="number" name="lead_time" min="0" max="999" value="0"> days</td>
                        <td><input type="text" name="arrival_date" size="10" style="text-align:center"
                                   class="form-control" id="date" placeholder="MM/DD/YYYY" value="08/01"></td>
                        <td><input type="number" name="stay_duration" min="0" max="365" value="3"> days</td>
                        <td><input type="number" name="adults" min="0" max="99" value="2"></td>
                        <td><input type="number" name="minors" min="0" max="99" value="0"></td>
                        <td><input type="checkbox" name="is_foreign" checked></td>
                        <td><select name="hotel_type">
                                <option value="city">City Hotel</option>
                                <option value="resort">Resort Hotel</option>
                            </select></td>
                        <td><select name="deposit_type">
                                <option value="no_deposit">No Deposit</option>
                                <option value="non_refund">Non Refund</option>
                                <option value="refundable">Refundable</option>
                            </select></td>
                        <td><select name="customer_type">
                                <option value="contract">Contract</option>
                                <option value="group">Group</option>
                                <option value="transient" selected>Transient</option>
                                <option value="transient_party">Transient-Party</option>
                            </select></td>
                    </tr>
                </tbody>
            </table>
            <label for="classifier">Classifier:</label>
            <select name="classifier" id="classifier">
                <option value="dt">Decision Tree</option>
                <option value="lgr">Logistic Regression</option>
                <option value="nb">Naive Bayes</option>
                <option value="rf">Random Forest</option>
            </select><br><br>
            <input type="submit" value="Predict!">
        </form>
        <p>
            <?php
                include("predict.php");
                if (!empty($_POST)) {
                    echo '<table class="table table-bordered" style="text-align:center; width:auto; float:left; margin:20px"><thead><tr><th colspan="2">Input</th></tr></thead><tbody>';
                    foreach ($_POST as $k => $v) {
                        echo "<tr><td>$k</td><td>$v</td></tr>";
                    }
                    echo '</tbody></table>';
                    echo '<table class="table table-bordered" style="text-align:center; width:auto; float:left; margin:20px"><thead><tr><th colspan="2">Output</th></tr></thead><tbody>';
                    $raw_outcome = str_replace("'", '"', get_prediction($_POST));
                    $outcome = json_decode($raw_outcome);
                    foreach ($outcome as $k => $v) {
                        echo "<tr><td>$k</td><td>$v</td></tr>";
                    }
                    echo '</tbody></table>';
                }
            ?>
        </p>
    </div>
</body>
</html>