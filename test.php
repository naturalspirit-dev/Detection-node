<?php
$input_dir = "result.jpg";
$output_dir = "result.png";
$python_script = "index.js";
$cmd = "node " . $python_script . " " . $input_dir . " " . $output_dir;
print_r($cmd);
exec($cmd, $output);
print_r("\n");
print_r($output);
?>
