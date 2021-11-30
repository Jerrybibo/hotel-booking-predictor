<?php
function get_prediction($post) {
    $post_json = base64_encode(json_encode($post));
//    $cmd = escapeshellcmd('cd /home/juhaozhang3_gmail_com/cs334; python3 predict.py ' . $post_json);
    $cmd = escapeshellcmd('python3 ../predict.py ' . $post_json);
    return shell_exec($cmd);
}