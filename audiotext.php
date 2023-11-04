<?php
$mysql_host = '127.0.0.1';
$mysql_user = 'root';
$mysql_pass = 'vpassword';
$conn = mysqli_connect($mysql_host, $mysql_user, $mysql_pass);
if ($conn && mysqli_select_db($conn, 'mosvotedb')) {
        //echo 'connection established successfully';
    ;
} else {
    die('connection failed');
}
?>
<?php
//require('sql_connect.php');
if (isset($_POST['c_number']) && isset($_POST['c_name']) && !empty($_POST['c_number']) && !empty($_POST['c_name'])) {
    save_vote(strtolower($_POST['c_name']), $_POST['c_value'], $_POST['c_number'],  $conn);
    exit;
}

/*if(isset($_POST['search_contact']) && !empty($_POST['search_contact'])){
        $name = strtolower($_POST['search_contact']);
        search_contact($name,$conn);
        exit;
    }*/
?>
<?php
function search_vote($name, $phone_no, $conn)
{
    $query = "select vote from mos_votes where name like '" . mysqli_real_escape_string($conn, $name) . "%' and phone_no like '" . mysqli_real_escape_string($conn, $phone_no) . "%'";
    $query_run = mysqli_query($conn, $query);
    if ($query_run) {
        if (mysqli_num_rows($query_run) == NULL) {
            echo 'No reslts found';
        } else {
            while ($query_row = mysqli_fetch_assoc($query_run)) {
                $phone_no = $query_row['phone no'];
                $full_name = mysqli_fetch_assoc(mysqli_query($conn, "select name from mos_votes where phone_no='" . mysqli_real_escape_string($conn, $phone_no) . "'"));
                #echo '<br>contact no of ' . $full_name['name'] . ' is ' . $phone_no;
            }
        }
    } else {
        echo '<br>' . mysqli_error($conn);
    }
}

function save_vote($name, $vvote, $number, $conn)
{
    $query = "insert into mos_votes(id, name, vote, phone_no) values(NULL,'" . mysqli_real_escape_string($conn, $name) . "','" . mysqli_real_escape_string($conn, $vvote) . "','" . mysqli_real_escape_string($conn, $number) . "')";
    if ($query_run = mysqli_query($conn, $query)) {
        echo '<br><br>vote saved!!!!';
    } else {
        echo '<br>' . mysqli_error($conn);
    }
}



?>
<html>

<head>
    <title>Gbe multilingual TTS System</title>
    <style>
        @import url('https://fonts.googleapis.com/css?family=Roboto:400,400i,700');

        body {
            font-family: 'Roboto', Arial, sans-serif;
        }

        td {
            max-width: 500px;
        }

        .text {
            font-style: italic;
            color: #666666;
            font-size: 11pt;
        }
    </style>
    <script type="text/javascript" src="./jquery-3.6.4.min.js"></script>
</head>

<body>
    <h1>Les audios Fongbe/Yoruba/Gungbe/Fongbe générés par le système plurilingue de synthèse vocale des langues GBE</h1>
    <p>
        Les exemples suivants sont générés à partir du modèle 16 kHz entraîné disponible sur notre
        <a href="https://github.com/bococharbel">page de projet</a>.
    </p>


    <fieldset>
        <legend>Téléphone</legend>
        <!--br>Votre numéro de téléphone :  -->
        <label for="sav_num">Votre numéro de téléphone </label>
        <input type="number" id="sav_num" name="c_number" min="10000000" max="99999999">
    </fieldset>



    <fieldset>
        <button type="submit" id='soumettre' value="Envoyer"></button>
        <!-- <br><input type="button" value="submit" onclick="insert()">-->
    </fieldset>

</body>
<footer></footer>
<script type="text/javascript">
    $(document).ready(function() {
        $("#soumettre").click(function() {
            //$("#testblock").hide();
            // $('.testimonial').each(function(i, obj) {
            //     //test
            // });
            // $('div[class="testimonial"]').each(function(index, item) {
            //     if (parseInt($(item).data('index')) > 2) {
            //         $(item).html('Testimonial ' + (index + 1) + ' by each loop');
            //     }
            // });
            $('.total').each(function(index, obj) {
                //you can use this to access the current item
                //$( this ).toggleClass( "example" );
                m_id = $(obj).id; //attr('id');//.prop('id')
                m_value = $(obj).children('span')[0].text(); //$('img', this)[0]
                $.post("index.php", {
                        c_name: m_id,
                        c_value: m_value,
                        c_number: $("#sav_num").value()
                    }, function() {
                        alert("Enregitrer avec succes");
                    })
                    .done(function() {
                        //alert("second success");
                    })
                    .fail(function() {
                        alert("error");
                    })
                    .always(function() {
                        //alert("finished");
                    });
            });
        });
    });
</script>
<script>
    function insert() {
        if (window.XMLHttpRequest) {
            xmlhttp = new XMLHttpRequest();
        } else {
            xmlhttp = new ActiveXObject('Microsoft.XMLHTTP');
        }
        xmlhttp.onreadystatechange = function() {
            if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
                document.getElementById('contact_saved').innerHTML = xmlhttp.responseText;
            }
        }
        parameters = 'c_number=' + document.getElementById('sav_num').value + '&c_name=' + document.getElementById('sav_nam').value;
        xmlhttp.open('POST', 'phonebook.php', true);
        xmlhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
        xmlhttp.send(parameters);
    }
</script>
<script>
    function search() {
        if (window.XMLHttpRequest) {
            xmlhttp = new XMLHttpRequest();
        } else {
            xmlhttp = new ActiveXObject('Microsoft.XMLHTTP');
        }
        xmlhttp.onreadystatechange = function() {
            if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
                document.getElementById('contact_saved').innerHTML = xmlhttp.responseText;
            }
        }
        parameters = 'search_contact=' + document.getElementById('search_num').value;
        xmlhttp.open('POST', 'phonebook.php', true);
        xmlhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
        xmlhttp.send(parameters);
    }
</script>

</html>