<!-- todo -->

<?php
    $result = passthru('python3 ../../foo.py' . json_encode($_POST));
    print($result);
?>

