<?php
$mysql_host = 'localhost'; //'127.0.0.1';
$mysql_user = 'id20661661_ttsuser';//root
$mysql_pass = 'd$uRa#U_2c5Q[qlK';//root
$conn = mysqli_connect($mysql_host, $mysql_user, $mysql_pass);
if ($conn && mysqli_select_db($conn, 'id20661661_ttsdb')) {
        //echo 'connection established successfully';
    ;
} else {
    die('connection failed');
}
?>
<?php
//require('sql_connect.php');
if (isset($_POST['c_number']) && isset($_POST['c_name']) && !empty($_POST['c_number']) && !empty($_POST['c_name'])) {
    //var_dump($_POST);
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
        json_response(200, "ENREGISTRE AVEC SUCCES"); //http_response_code(200); //echo '<br><br>vote saved!!!!';
    } else {
        json_response(500, "ERROR " . mysqli_error($conn)); //http_response_code(500); //echo '<br>' . mysqli_error($conn);
    }
}

function json_response($code = 200, $message = null)
{
    // clear the old headers
    header_remove();
    // set the actual code
    http_response_code($code);
    // set the header to make sure cache is forced
    header("Cache-Control: no-transform,public,max-age=300,s-maxage=900");
    // treat this as json
    header('Content-Type: application/json');
    $status = array(
        200 => '200 OK',
        400 => '400 Bad Request',
        422 => 'Unprocessable Entity',
        500 => '500 Internal Server Error'
    );
    // ok, validation error, or failure
    header('Status: ' . $status[$code]);
    // return the encoded json
    echo json_encode(array(
        'status' => $code < 300, // success or not?
        'message' => $message
    ));
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

        .total .text {
            background-color: #008000c9;
            color: black;
            font-weight: 200%;
            padding: 10 20 10 20;
        }

        .content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 500px;
            height: 200px;
            text-align: center;
            background-color: #e8eae6;
            box-sizing: border-box;
            padding: 10px;
            z-index: 100;
            display: none;
            /*to hide popup initially*/
        }

        .close-btn {
            position: absolute;
            right: 20px;
            top: 15px;
            background-color: black;
            color: white;
            border-radius: 50%;
            padding: 4px;
        }
    </style>
    <script type="text/javascript" src="./jquery-3.6.4.min.js"></script>
</head>

<body>
    <div class="content">
        <div onclick="togglePopup()" class="close-btn">
            ×
        </div>
        <h3>Informations</h3>

        <p id='content_message' data-message='' style="color:green">

        </p>
    </div>
    <h1>Les audios Fongbe/Yoruba/Gungbe/Fongbe générés par le système plurilingue de synthèse vocale des langues GBE</h1>
    <p>
        Les exemples suivants sont générés à partir du modèle 16 kHz entraîné disponible sur notre
        <a href="https://github.com/bococharbel">page de projet</a>.
    </p>


    <fieldset>
        <legend>Téléphone</legend>
        <!--br>Votre numéro de téléphone :  -->
        <label for="sav_num">Votre numéro de téléphone </label>
        <input type="phone" id="sav_num" name="c_number" minlength="8" maxlength="8" placeholder="Numero de telephone" pattern="[1-9]{1}[0-9]{7}" min="10000000" max="9999999999999">
    </fieldset>


    <div id='sumessage' style="background: #fafafa; color:green; border: 1px solid #fafafa; width:100%; display:none; margin:10 10 10 10;"></div>


    <!-- --------------#------------- start tables group --------------#------------- -->

    <H1> VQ </H1>


    <H2> fongbe </H2>


    <H3> choosen </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> KANLIN DÍDÁ WƐ YĚ NYÍ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_00027.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___01_Tite________FONBSBN1DA_00027_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> avivɔ kpódó yǒzo kpó ná tíin, .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/biblebibeme_8_29.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/biblebibeme_8_29_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> xo vέ sìn yɔ̀ bε sɔ́ awìnyán ɖó ado jí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/ph17n2.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/ph17n2_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> XÓ NǓGBÓ ƉOKPÓ NƐ́. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___03_Tite________FONBSBN1DA_0008.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___03_Tite________FONBSBN1DA_0008_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> nùɖé ɖó nukún nyɔ́ hú nùɖé má ɖó nukún. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/ph17n3.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/ph17n3_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> YĚ NÁ NƆ́ ƉƆ MƐ NU Ǎ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________FONBSBN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___02_Tite________FONBSBN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> ÉTƐ́ UN KA NÁ LƐ́ ƉƆ? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___11_Hebreux_____FONBSBN1DA_00057.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B19___11_Hebreux_____FONBSBN1DA_00057_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> YĚ NÁ NYÍ AHANNUMÚNƆ Ǎ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________FONBSBN1DA_00008.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___02_Tite________FONBSBN1DA_00008_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ; É ƉÓ NÁ NƆ WA NǓ ƉÓ JLƐ̌ JÍ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_0013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___01_Tite________FONBSBN1DA_0013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> e hun hɔn ɔ gblawun .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/inconnu_fongbe_corp015_014.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/inconnu_fongbe_corp015_014_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> VǏ TƆN LƐ́Ɛ ƉÓ NÁ ƉI NǓ NÚ KLÍSU; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___01_Tite________FONBSBN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> È NYI AWǏNNYAGLO DÓ MƐƉÉ LƐ́Ɛ KÁKÁ BƆ YĚ KÚ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___11_Hebreux_____FONBSBN1DA_00070.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B19___11_Hebreux_____FONBSBN1DA_00070_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> ASI ƉOKPÓ GÉÉ WƐ É ƉÓ NÁ ƉÓ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/fongbe/choosen/B17___01_Tite________FONBSBN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_fongbe_choosen" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H2> yoruba </H2>


    <H3> choosen </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> Ó figbe sẹ́nu pariwo. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_03034_01847424174.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_03034_01847424174_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> Mo ò dára nínú eré bọ́ọ̀lù gbígbá. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_06136_00852892464.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_06136_00852892464_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> Ta ló gbé aṣọ tútù sí ibí? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07508_00510677721.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_07508_00510677721_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> Ó fẹ́ pẹ́ láyé ju bí ó ti yẹ lọ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07049_01725771540.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_07049_01725771540_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> Magí ti já ọbẹ̀ yẹn. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_05223_01066585617.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_05223_01066585617_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> Aya rere lọ́dẹ̀dẹ̀ ọkọ ni Ṣọlá. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_09697_00469094337.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_09697_00469094337_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> Ọ̀gá akọrin wa féraǹ mi gidi. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_08784_01362090520.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_08784_01362090520_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> Wàhálà ti mo fi ara ṣe pò lénìí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_03349_00034880533.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_03349_00034880533_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> Ta ló gbé ọmọ jù sórí àkìtàn? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_04310_01265456913.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_04310_01265456913_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> Kò sẹ́ni tó mọ Ifá tán. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_03349_01957803301.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_03349_01957803301_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> Ìgbákọ tí mo fẹ́ lò dà? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07505_01439149460.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_07505_01439149460_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> Wàhálà lọ́tùún, rògbòdìyàn lósì. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_09697_01342949732.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_09697_01342949732_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> Imú ọmọ ìkókó náà dọ̀tí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_09697_00764365820.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_09697_00764365820_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> Àwọn mélòó ló wà nínú yàrá yìíi? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_08784_00741374779.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_08784_00741374779_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> Ẹ fi ṣàkì mẹ́ta sórí ìrẹsì àkọ́kọ́. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_02484_00968759546.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_02484_00968759546_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> Lára àwọn agbẹjọ́rò àgbà ni Fálànà wà. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_05223_00533851592.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_05223_00533851592_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> Ìjọba yóò fi kún owó àwọn òṣìṣẹ́. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07505_01448838967.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_07505_01448838967_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> Ìyálẹ̀ta la wà báyìí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_00295_00564596981.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_00295_00564596981_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 19 </span></td>
                <td><span class="text"> Ki lo kùn sí ètè? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_02436_02103674726.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_02436_02103674726_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 20 </span></td>
                <td><span class="text"> A ti gé igi kan lulẹ̀. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07049_01581353881.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/yoruba/choosen/yof_07049_01581353881_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_yoruba_choosen" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H2> gengbe </H2>


    <H3> 1h </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MÙ JI BE WÒA WƆ̀ Ɛ̀ KUDO ÈJÌ FÀA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00015.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00015_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> EYE MÙ KÃ ÀTÃŊU, LÈ ÀPE ÀDƆ̀MÈZE MÈ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___03_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> VƆ̀ FIFÌA, E ƉÈ ÀLÈ NA ÈNYÈ KUDO ÈWÒ CÃ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> EKÈA YE NYI ÈNU KÈ ŊƆ̀ŊLƆ̃ A LE GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___03_Hebreux_____GEJBS2N2DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> SÃA, MU ƉÈ ÀLÈ ƉEKPE NA WÒ Ò; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> MÙ TRƆ Ɛ̀ LE ƉOƉA WÒ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> WÒ NYI ÈVI NYÈ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___05_Hebreux_____GEJBS2N2DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> E SƆNA ÈNU NANAWO NANA, SANAVƆ ƉO ÈVUƐ̃ TA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___05_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ÀLO ÀGBÈTƆ BE ÈVI, NE WÒA KÃMA NA Ɛ̀ Ò? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___02_Hebreux_____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> NUKƐ ÀGBÈTƆ NYI, NE WÒA ƉO ŊÙKUVI E JI? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___02_Hebreux_____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> EYE E GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___02_Hebreux_____GEJBS2N2DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> MIA TƆHONƆ̀ YESÙ KRISTÒ BE MÀJE NE NƆ̀ KU MÌ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00030.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00030_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> MU LA ŊLƆ̀BE ÈDƆ KÈ MÌ LE WƆ̀A Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> EKÈA YE MI LA WƆ̀ NE MAWU ƉÈ ÈMƆA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00006.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00006_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> LÈ ÀPE ÀDƆ̀MÈZE MÈA, MÙ KÃ ÀTÃŊU BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___04_Hebreux_____GEJBS2N2DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___04_Hebreux_____GEJBS2N2DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> ÈNYÈ PAOLÒ ŊUTƆ YE SƆ ÀPE ÀLƆ̀ ŊLƆ̀ EKÈA BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00023.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00023_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> MI MU LA GBÀ KƆ ÀŊƐ̀A BE GƆ̃MÈJÈJE BE ÈNYÀA YÌJI Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> MIA ƉÈ ÀSI LÈ XƆ̀SÈA BE KPAKPLA CUCUGBÃA ŊUTI. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gengbe_1h" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 2h </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MÙ JI BE WÒA WƆ̀ Ɛ̀ KUDO ÈJÌ FÀA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00015.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B18___01_Philemon____GEJBS2N2DA_00015_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> EYE MÙ KÃ ÀTÃŊU, LÈ ÀPE ÀDƆ̀MÈZE MÈ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___03_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> VƆ̀ FIFÌA, E ƉÈ ÀLÈ NA ÈNYÈ KUDO ÈWÒ CÃ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B18___01_Philemon____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> EKÈA YE NYI ÈNU KÈ ŊƆ̀ŊLƆ̃ A LE GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___03_Hebreux_____GEJBS2N2DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> SÃA, MU ƉÈ ÀLÈ ƉEKPE NA WÒ Ò; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B18___01_Philemon____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> MÙ TRƆ Ɛ̀ LE ƉOƉA WÒ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B18___01_Philemon____GEJBS2N2DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> WÒ NYI ÈVI NYÈ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___05_Hebreux_____GEJBS2N2DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> E SƆNA ÈNU NANAWO NANA, SANAVƆ ƉO ÈVUƐ̃ TA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___05_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ÀLO ÀGBÈTƆ BE ÈVI, NE WÒA KÃMA NA Ɛ̀ Ò? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___02_Hebreux_____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> NUKƐ ÀGBÈTƆ NYI, NE WÒA ƉO ŊÙKUVI E JI? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___02_Hebreux_____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> EYE E GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___02_Hebreux_____GEJBS2N2DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> MIA TƆHONƆ̀ YESÙ KRISTÒ BE MÀJE NE NƆ̀ KU MÌ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00030.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B18___01_Philemon____GEJBS2N2DA_00030_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> MU LA ŊLƆ̀BE ÈDƆ KÈ MÌ LE WƆ̀A Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___06_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> EKÈA YE MI LA WƆ̀ NE MAWU ƉÈ ÈMƆA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00006.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___06_Hebreux_____GEJBS2N2DA_00006_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> LÈ ÀPE ÀDƆ̀MÈZE MÈA, MÙ KÃ ÀTÃŊU BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___04_Hebreux_____GEJBS2N2DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___04_Hebreux_____GEJBS2N2DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> ÈNYÈ PAOLÒ ŊUTƆ YE SƆ ÀPE ÀLƆ̀ ŊLƆ̀ EKÈA BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00023.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B18___01_Philemon____GEJBS2N2DA_00023_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> MI MU LA GBÀ KƆ ÀŊƐ̀A BE GƆ̃MÈJÈJE BE ÈNYÀA YÌJI Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___06_Hebreux_____GEJBS2N2DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> MIA ƉÈ ÀSI LÈ XƆ̀SÈA BE KPAKPLA CUCUGBÃA ŊUTI. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/2h/B19___06_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gengbe_2h" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 15mn </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MÙ JI BE WÒA WƆ̀ Ɛ̀ KUDO ÈJÌ FÀA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00015.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B18___01_Philemon____GEJBS2N2DA_00015_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> EYE MÙ KÃ ÀTÃŊU, LÈ ÀPE ÀDƆ̀MÈZE MÈ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___03_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> VƆ̀ FIFÌA, E ƉÈ ÀLÈ NA ÈNYÈ KUDO ÈWÒ CÃ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B18___01_Philemon____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> EKÈA YE NYI ÈNU KÈ ŊƆ̀ŊLƆ̃ A LE GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___03_Hebreux_____GEJBS2N2DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> SÃA, MU ƉÈ ÀLÈ ƉEKPE NA WÒ Ò; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B18___01_Philemon____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> MÙ TRƆ Ɛ̀ LE ƉOƉA WÒ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B18___01_Philemon____GEJBS2N2DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> WÒ NYI ÈVI NYÈ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___05_Hebreux_____GEJBS2N2DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> E SƆNA ÈNU NANAWO NANA, SANAVƆ ƉO ÈVUƐ̃ TA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___05_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ÀLO ÀGBÈTƆ BE ÈVI, NE WÒA KÃMA NA Ɛ̀ Ò? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___02_Hebreux_____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> NUKƐ ÀGBÈTƆ NYI, NE WÒA ƉO ŊÙKUVI E JI? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___02_Hebreux_____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> EYE E GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___02_Hebreux_____GEJBS2N2DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> MIA TƆHONƆ̀ YESÙ KRISTÒ BE MÀJE NE NƆ̀ KU MÌ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00030.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B18___01_Philemon____GEJBS2N2DA_00030_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> MU LA ŊLƆ̀BE ÈDƆ KÈ MÌ LE WƆ̀A Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___06_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> EKÈA YE MI LA WƆ̀ NE MAWU ƉÈ ÈMƆA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00006.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___06_Hebreux_____GEJBS2N2DA_00006_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> LÈ ÀPE ÀDƆ̀MÈZE MÈA, MÙ KÃ ÀTÃŊU BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___04_Hebreux_____GEJBS2N2DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___04_Hebreux_____GEJBS2N2DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> ÈNYÈ PAOLÒ ŊUTƆ YE SƆ ÀPE ÀLƆ̀ ŊLƆ̀ EKÈA BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00023.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B18___01_Philemon____GEJBS2N2DA_00023_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> MI MU LA GBÀ KƆ ÀŊƐ̀A BE GƆ̃MÈJÈJE BE ÈNYÀA YÌJI Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___06_Hebreux_____GEJBS2N2DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> MIA ƉÈ ÀSI LÈ XƆ̀SÈA BE KPAKPLA CUCUGBÃA ŊUTI. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/15mn/B19___06_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gengbe_15mn" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 30mn </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MÙ JI BE WÒA WƆ̀ Ɛ̀ KUDO ÈJÌ FÀA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00015.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00015_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> EYE MÙ KÃ ÀTÃŊU, LÈ ÀPE ÀDƆ̀MÈZE MÈ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___03_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> VƆ̀ FIFÌA, E ƉÈ ÀLÈ NA ÈNYÈ KUDO ÈWÒ CÃ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> EKÈA YE NYI ÈNU KÈ ŊƆ̀ŊLƆ̃ A LE GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___03_Hebreux_____GEJBS2N2DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> SÃA, MU ƉÈ ÀLÈ ƉEKPE NA WÒ Ò; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> MÙ TRƆ Ɛ̀ LE ƉOƉA WÒ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> WÒ NYI ÈVI NYÈ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___05_Hebreux_____GEJBS2N2DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> E SƆNA ÈNU NANAWO NANA, SANAVƆ ƉO ÈVUƐ̃ TA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___05_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ÀLO ÀGBÈTƆ BE ÈVI, NE WÒA KÃMA NA Ɛ̀ Ò? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___02_Hebreux_____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> NUKƐ ÀGBÈTƆ NYI, NE WÒA ƉO ŊÙKUVI E JI? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___02_Hebreux_____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> EYE E GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___02_Hebreux_____GEJBS2N2DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> MIA TƆHONƆ̀ YESÙ KRISTÒ BE MÀJE NE NƆ̀ KU MÌ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00030.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00030_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> MU LA ŊLƆ̀BE ÈDƆ KÈ MÌ LE WƆ̀A Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> EKÈA YE MI LA WƆ̀ NE MAWU ƉÈ ÈMƆA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00006.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00006_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> LÈ ÀPE ÀDƆ̀MÈZE MÈA, MÙ KÃ ÀTÃŊU BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___04_Hebreux_____GEJBS2N2DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___04_Hebreux_____GEJBS2N2DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> ÈNYÈ PAOLÒ ŊUTƆ YE SƆ ÀPE ÀLƆ̀ ŊLƆ̀ EKÈA BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00023.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00023_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> MI MU LA GBÀ KƆ ÀŊƐ̀A BE GƆ̃MÈJÈJE BE ÈNYÀA YÌJI Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> MIA ƉÈ ÀSI LÈ XƆ̀SÈA BE KPAKPLA CUCUGBÃA ŊUTI. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gengbe_30mn" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H2> gungbe </H2>


    <H3> 1h </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MALKU, ALISTALKU, DEMA, LUKU, AZÓNWÀTỌGBÉ ṢIE LẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00025.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00025_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> HIẸ HẸN ẸN WHÈ PẸVÌDE HÙ ANGELI LẸ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GUWNVSN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B19___02_Hebreux_____GUWNVSN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> MIỌNHOMẸNA MI TO OKLUNỌ MẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00021.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00021_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> HA DAGBEWÀ TOWE MA NA DO YIN NUMANANỌMAWÀ TỌN, ADAVO OJLO TỌN. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> ṢIGBA YEN MA YLO NADO WA NÚDE TO AYIHA TOWE MAYỌNẸN GODO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> ENẸWUTU NA HIẸ NI YÍ I, ENẸ WẸ, OVI OHÒ ṢIE MẸ TỌN: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> MẸHE YẸN SỌ GÒ DOHLAN; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> ENẸWUTU EYÍN HIẸ LẸN MI DO OHA MẸ, BO YÍ I DI YẸNLỌSU. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> YẸN PAULU KO YÍ ALỌ ṢIE TITI DO WLAN ẸN, YẸN NA SÚ I: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00018.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00018_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> ṢIGBA MẸHE DÓ ONÚ POPO WE JIWHEYẸWHE. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B19___03_Hebreux_____GUWNVSN1DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> NA YÈ HẸN MADOGÁN HE LÉDO EWLOSU GA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GUWNVSN1DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B19___05_Hebreux_____GUWNVSN1DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> HE MA NỌ DO DÓDÓ DAGBE LẸPO HIA; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B17___02_Tite________GUWNVSN1DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> NA MẸDE DALI WẸ YÈ GBỌN DO DÓ OHÒ LẸ DOPODOPO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00004.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B19___03_Hebreux_____GUWNVSN1DA_00004_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> BO SỌ NỌ DAGBÈDOGBÈMẸ BLO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/1h/B17___02_Tite________GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gungbe_1h" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 2h </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MALKU, ALISTALKU, DEMA, LUKU, AZÓNWÀTỌGBÉ ṢIE LẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00025.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00025_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> HIẸ HẸN ẸN WHÈ PẸVÌDE HÙ ANGELI LẸ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GUWNVSN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B19___02_Hebreux_____GUWNVSN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> MIỌNHOMẸNA MI TO OKLUNỌ MẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00021.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00021_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> HA DAGBEWÀ TOWE MA NA DO YIN NUMANANỌMAWÀ TỌN, ADAVO OJLO TỌN. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> ṢIGBA YEN MA YLO NADO WA NÚDE TO AYIHA TOWE MAYỌNẸN GODO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> ENẸWUTU NA HIẸ NI YÍ I, ENẸ WẸ, OVI OHÒ ṢIE MẸ TỌN: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> MẸHE YẸN SỌ GÒ DOHLAN; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> ENẸWUTU EYÍN HIẸ LẸN MI DO OHA MẸ, BO YÍ I DI YẸNLỌSU. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> YẸN PAULU KO YÍ ALỌ ṢIE TITI DO WLAN ẸN, YẸN NA SÚ I: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00018.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B18___01_Philemon____GUWNVSN1DA_00018_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> ṢIGBA MẸHE DÓ ONÚ POPO WE JIWHEYẸWHE. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B19___03_Hebreux_____GUWNVSN1DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> NA YÈ HẸN MADOGÁN HE LÉDO EWLOSU GA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GUWNVSN1DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B19___05_Hebreux_____GUWNVSN1DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> HE MA NỌ DO DÓDÓ DAGBE LẸPO HIA; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B17___02_Tite________GUWNVSN1DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> NA MẸDE DALI WẸ YÈ GBỌN DO DÓ OHÒ LẸ DOPODOPO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00004.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B19___03_Hebreux_____GUWNVSN1DA_00004_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> BO SỌ NỌ DAGBÈDOGBÈMẸ BLO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/2h/B17___02_Tite________GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_2h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gungbe_2h" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 15mn </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MALKU, ALISTALKU, DEMA, LUKU, AZÓNWÀTỌGBÉ ṢIE LẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00025.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00025_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> HIẸ HẸN ẸN WHÈ PẸVÌDE HÙ ANGELI LẸ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GUWNVSN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B19___02_Hebreux_____GUWNVSN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> MIỌNHOMẸNA MI TO OKLUNỌ MẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00021.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00021_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> HA DAGBEWÀ TOWE MA NA DO YIN NUMANANỌMAWÀ TỌN, ADAVO OJLO TỌN. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> ṢIGBA YEN MA YLO NADO WA NÚDE TO AYIHA TOWE MAYỌNẸN GODO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> ENẸWUTU NA HIẸ NI YÍ I, ENẸ WẸ, OVI OHÒ ṢIE MẸ TỌN: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> MẸHE YẸN SỌ GÒ DOHLAN; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> ENẸWUTU EYÍN HIẸ LẸN MI DO OHA MẸ, BO YÍ I DI YẸNLỌSU. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> YẸN PAULU KO YÍ ALỌ ṢIE TITI DO WLAN ẸN, YẸN NA SÚ I: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00018.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B18___01_Philemon____GUWNVSN1DA_00018_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> ṢIGBA MẸHE DÓ ONÚ POPO WE JIWHEYẸWHE. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B19___03_Hebreux_____GUWNVSN1DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> NA YÈ HẸN MADOGÁN HE LÉDO EWLOSU GA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GUWNVSN1DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B19___05_Hebreux_____GUWNVSN1DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> HE MA NỌ DO DÓDÓ DAGBE LẸPO HIA; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B17___02_Tite________GUWNVSN1DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> NA MẸDE DALI WẸ YÈ GBỌN DO DÓ OHÒ LẸ DOPODOPO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00004.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B19___03_Hebreux_____GUWNVSN1DA_00004_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> BO SỌ NỌ DAGBÈDOGBÈMẸ BLO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/15mn/B17___02_Tite________GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_15mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gungbe_15mn" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 30mn </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MALKU, ALISTALKU, DEMA, LUKU, AZÓNWÀTỌGBÉ ṢIE LẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00025.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00025_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> HIẸ HẸN ẸN WHÈ PẸVÌDE HÙ ANGELI LẸ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GUWNVSN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B19___02_Hebreux_____GUWNVSN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> MIỌNHOMẸNA MI TO OKLUNỌ MẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00021.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00021_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> HA DAGBEWÀ TOWE MA NA DO YIN NUMANANỌMAWÀ TỌN, ADAVO OJLO TỌN. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> ṢIGBA YEN MA YLO NADO WA NÚDE TO AYIHA TOWE MAYỌNẸN GODO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> ENẸWUTU NA HIẸ NI YÍ I, ENẸ WẸ, OVI OHÒ ṢIE MẸ TỌN: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> MẸHE YẸN SỌ GÒ DOHLAN; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> ENẸWUTU EYÍN HIẸ LẸN MI DO OHA MẸ, BO YÍ I DI YẸNLỌSU. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> YẸN PAULU KO YÍ ALỌ ṢIE TITI DO WLAN ẸN, YẸN NA SÚ I: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00018.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00018_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> ṢIGBA MẸHE DÓ ONÚ POPO WE JIWHEYẸWHE. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B19___03_Hebreux_____GUWNVSN1DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> NA YÈ HẸN MADOGÁN HE LÉDO EWLOSU GA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GUWNVSN1DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B19___05_Hebreux_____GUWNVSN1DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> HE MA NỌ DO DÓDÓ DAGBE LẸPO HIA; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B17___02_Tite________GUWNVSN1DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> NA MẸDE DALI WẸ YÈ GBỌN DO DÓ OHÒ LẸ DOPODOPO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00004.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B19___03_Hebreux_____GUWNVSN1DA_00004_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> BO SỌ NỌ DAGBÈDOGBÈMẸ BLO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/VQ/gungbe/30mn/B17___02_Tite________GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='VQ_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalVQ_gungbe_30mn" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H1> no-vq </H1>


    <H2> fongbe </H2>


    <H3> choosen </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> KANLIN DÍDÁ WƐ YĚ NYÍ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_00027.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___01_Tite________FONBSBN1DA_00027_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> avivɔ kpódó yǒzo kpó ná tíin, .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/biblebibeme_8_29.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/biblebibeme_8_29_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> xo vέ sìn yɔ̀ bε sɔ́ awìnyán ɖó ado jí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/ph17n2.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/ph17n2_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> XÓ NǓGBÓ ƉOKPÓ NƐ́. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___03_Tite________FONBSBN1DA_0008.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___03_Tite________FONBSBN1DA_0008_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> nùɖé ɖó nukún nyɔ́ hú nùɖé má ɖó nukún. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/ph17n3.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/ph17n3_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> YĚ NÁ NƆ́ ƉƆ MƐ NU Ǎ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________FONBSBN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___02_Tite________FONBSBN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> ÉTƐ́ UN KA NÁ LƐ́ ƉƆ? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___11_Hebreux_____FONBSBN1DA_00057.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B19___11_Hebreux_____FONBSBN1DA_00057_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> YĚ NÁ NYÍ AHANNUMÚNƆ Ǎ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________FONBSBN1DA_00008.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___02_Tite________FONBSBN1DA_00008_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ; É ƉÓ NÁ NƆ WA NǓ ƉÓ JLƐ̌ JÍ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_0013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___01_Tite________FONBSBN1DA_0013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> e hun hɔn ɔ gblawun .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/inconnu_fongbe_corp015_014.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/inconnu_fongbe_corp015_014_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> VǏ TƆN LƐ́Ɛ ƉÓ NÁ ƉI NǓ NÚ KLÍSU; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___01_Tite________FONBSBN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> È NYI AWǏNNYAGLO DÓ MƐƉÉ LƐ́Ɛ KÁKÁ BƆ YĚ KÚ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___11_Hebreux_____FONBSBN1DA_00070.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B19___11_Hebreux_____FONBSBN1DA_00070_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> ASI ƉOKPÓ GÉÉ WƐ É ƉÓ NÁ ƉÓ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___01_Tite________FONBSBN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/fongbe/choosen/B17___01_Tite________FONBSBN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_fongbe_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalno_vq_fongbe_choosen" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H2> yoruba </H2>


    <H3> choosen </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> Ó figbe sẹ́nu pariwo. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_03034_01847424174.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_03034_01847424174_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> Mo ò dára nínú eré bọ́ọ̀lù gbígbá. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_06136_00852892464.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_06136_00852892464_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> Ta ló gbé aṣọ tútù sí ibí? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07508_00510677721.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_07508_00510677721_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> Ó fẹ́ pẹ́ láyé ju bí ó ti yẹ lọ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07049_01725771540.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_07049_01725771540_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> Magí ti já ọbẹ̀ yẹn. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_05223_01066585617.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_05223_01066585617_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> Aya rere lọ́dẹ̀dẹ̀ ọkọ ni Ṣọlá. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_09697_00469094337.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_09697_00469094337_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> Ọ̀gá akọrin wa féraǹ mi gidi. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_08784_01362090520.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_08784_01362090520_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> Wàhálà ti mo fi ara ṣe pò lénìí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_03349_00034880533.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_03349_00034880533_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> Ta ló gbé ọmọ jù sórí àkìtàn? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_04310_01265456913.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_04310_01265456913_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> Kò sẹ́ni tó mọ Ifá tán. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_03349_01957803301.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_03349_01957803301_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> Ìgbákọ tí mo fẹ́ lò dà? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07505_01439149460.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_07505_01439149460_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> Wàhálà lọ́tùún, rògbòdìyàn lósì. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_09697_01342949732.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_09697_01342949732_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> Imú ọmọ ìkókó náà dọ̀tí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_09697_00764365820.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_09697_00764365820_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> Àwọn mélòó ló wà nínú yàrá yìíi? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_08784_00741374779.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_08784_00741374779_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> Ẹ fi ṣàkì mẹ́ta sórí ìrẹsì àkọ́kọ́. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_02484_00968759546.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_02484_00968759546_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> Lára àwọn agbẹjọ́rò àgbà ni Fálànà wà. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_05223_00533851592.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_05223_00533851592_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> Ìjọba yóò fi kún owó àwọn òṣìṣẹ́. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07505_01448838967.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_07505_01448838967_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> Ìyálẹ̀ta la wà báyìí. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_00295_00564596981.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_00295_00564596981_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 19 </span></td>
                <td><span class="text"> Ki lo kùn sí ètè? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_02436_02103674726.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_02436_02103674726_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 20 </span></td>
                <td><span class="text"> A ti gé igi kan lulẹ̀. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/yof_07049_01581353881.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/yoruba/choosen/yof_07049_01581353881_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_yoruba_choosen' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalno_vq_yoruba_choosen" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H2> gengbe </H2>


    <H3> 1h </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MÙ JI BE WÒA WƆ̀ Ɛ̀ KUDO ÈJÌ FÀA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00015.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00015_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> EYE MÙ KÃ ÀTÃŊU, LÈ ÀPE ÀDƆ̀MÈZE MÈ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___03_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> VƆ̀ FIFÌA, E ƉÈ ÀLÈ NA ÈNYÈ KUDO ÈWÒ CÃ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> EKÈA YE NYI ÈNU KÈ ŊƆ̀ŊLƆ̃ A LE GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___03_Hebreux_____GEJBS2N2DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> SÃA, MU ƉÈ ÀLÈ ƉEKPE NA WÒ Ò; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> MÙ TRƆ Ɛ̀ LE ƉOƉA WÒ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> WÒ NYI ÈVI NYÈ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___05_Hebreux_____GEJBS2N2DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> E SƆNA ÈNU NANAWO NANA, SANAVƆ ƉO ÈVUƐ̃ TA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___05_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ÀLO ÀGBÈTƆ BE ÈVI, NE WÒA KÃMA NA Ɛ̀ Ò? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___02_Hebreux_____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> NUKƐ ÀGBÈTƆ NYI, NE WÒA ƉO ŊÙKUVI E JI? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___02_Hebreux_____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> EYE E GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___02_Hebreux_____GEJBS2N2DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> MIA TƆHONƆ̀ YESÙ KRISTÒ BE MÀJE NE NƆ̀ KU MÌ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00030.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00030_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> MU LA ŊLƆ̀BE ÈDƆ KÈ MÌ LE WƆ̀A Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> EKÈA YE MI LA WƆ̀ NE MAWU ƉÈ ÈMƆA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00006.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00006_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> LÈ ÀPE ÀDƆ̀MÈZE MÈA, MÙ KÃ ÀTÃŊU BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___04_Hebreux_____GEJBS2N2DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___04_Hebreux_____GEJBS2N2DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> ÈNYÈ PAOLÒ ŊUTƆ YE SƆ ÀPE ÀLƆ̀ ŊLƆ̀ EKÈA BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00023.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B18___01_Philemon____GEJBS2N2DA_00023_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> MI MU LA GBÀ KƆ ÀŊƐ̀A BE GƆ̃MÈJÈJE BE ÈNYÀA YÌJI Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> MIA ƉÈ ÀSI LÈ XƆ̀SÈA BE KPAKPLA CUCUGBÃA ŊUTI. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/1h/B19___06_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalno_vq_gengbe_1h" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 30mn </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MÙ JI BE WÒA WƆ̀ Ɛ̀ KUDO ÈJÌ FÀA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00015.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00015_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> EYE MÙ KÃ ÀTÃŊU, LÈ ÀPE ÀDƆ̀MÈZE MÈ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___03_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> VƆ̀ FIFÌA, E ƉÈ ÀLÈ NA ÈNYÈ KUDO ÈWÒ CÃ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> EKÈA YE NYI ÈNU KÈ ŊƆ̀ŊLƆ̃ A LE GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GEJBS2N2DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___03_Hebreux_____GEJBS2N2DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> SÃA, MU ƉÈ ÀLÈ ƉEKPE NA WÒ Ò; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> MÙ TRƆ Ɛ̀ LE ƉOƉA WÒ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> WÒ NYI ÈVI NYÈ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___05_Hebreux_____GEJBS2N2DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> E SƆNA ÈNU NANAWO NANA, SANAVƆ ƉO ÈVUƐ̃ TA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___05_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> ÀLO ÀGBÈTƆ BE ÈVI, NE WÒA KÃMA NA Ɛ̀ Ò? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___02_Hebreux_____GEJBS2N2DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> NUKƐ ÀGBÈTƆ NYI, NE WÒA ƉO ŊÙKUVI E JI? .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___02_Hebreux_____GEJBS2N2DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> EYE E GBLƆ̃ BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GEJBS2N2DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___02_Hebreux_____GEJBS2N2DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> MIA TƆHONƆ̀ YESÙ KRISTÒ BE MÀJE NE NƆ̀ KU MÌ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00030.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00030_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> MU LA ŊLƆ̀BE ÈDƆ KÈ MÌ LE WƆ̀A Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> EKÈA YE MI LA WƆ̀ NE MAWU ƉÈ ÈMƆA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00006.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00006_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 15 </span></td>
                <td><span class="text"> LÈ ÀPE ÀDƆ̀MÈZE MÈA, MÙ KÃ ÀTÃŊU BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___04_Hebreux_____GEJBS2N2DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___04_Hebreux_____GEJBS2N2DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 16 </span></td>
                <td><span class="text"> ÈNYÈ PAOLÒ ŊUTƆ YE SƆ ÀPE ÀLƆ̀ ŊLƆ̀ EKÈA BE: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GEJBS2N2DA_00023.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B18___01_Philemon____GEJBS2N2DA_00023_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 17 </span></td>
                <td><span class="text"> MI MU LA GBÀ KƆ ÀŊƐ̀A BE GƆ̃MÈJÈJE BE ÈNYÀA YÌJI Ò. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 18 </span></td>
                <td><span class="text"> MIA ƉÈ ÀSI LÈ XƆ̀SÈA BE KPAKPLA CUCUGBÃA ŊUTI. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___06_Hebreux_____GEJBS2N2DA_00002.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gengbe/30mn/B19___06_Hebreux_____GEJBS2N2DA_00002_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gengbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalno_vq_gengbe_30mn" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H2> gungbe </H2>


    <H3> 1h </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MALKU, ALISTALKU, DEMA, LUKU, AZÓNWÀTỌGBÉ ṢIE LẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00025.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00025_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> HIẸ HẸN ẸN WHÈ PẸVÌDE HÙ ANGELI LẸ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GUWNVSN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B19___02_Hebreux_____GUWNVSN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> MIỌNHOMẸNA MI TO OKLUNỌ MẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00021.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00021_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> HA DAGBEWÀ TOWE MA NA DO YIN NUMANANỌMAWÀ TỌN, ADAVO OJLO TỌN. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> ṢIGBA YEN MA YLO NADO WA NÚDE TO AYIHA TOWE MAYỌNẸN GODO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> ENẸWUTU NA HIẸ NI YÍ I, ENẸ WẸ, OVI OHÒ ṢIE MẸ TỌN: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> MẸHE YẸN SỌ GÒ DOHLAN; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> ENẸWUTU EYÍN HIẸ LẸN MI DO OHA MẸ, BO YÍ I DI YẸNLỌSU. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> YẸN PAULU KO YÍ ALỌ ṢIE TITI DO WLAN ẸN, YẸN NA SÚ I: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00018.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B18___01_Philemon____GUWNVSN1DA_00018_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> ṢIGBA MẸHE DÓ ONÚ POPO WE JIWHEYẸWHE. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B19___03_Hebreux_____GUWNVSN1DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> NA YÈ HẸN MADOGÁN HE LÉDO EWLOSU GA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GUWNVSN1DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B19___05_Hebreux_____GUWNVSN1DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> HE MA NỌ DO DÓDÓ DAGBE LẸPO HIA; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B17___02_Tite________GUWNVSN1DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> NA MẸDE DALI WẸ YÈ GBỌN DO DÓ OHÒ LẸ DOPODOPO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00004.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B19___03_Hebreux_____GUWNVSN1DA_00004_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> BO SỌ NỌ DAGBÈDOGBÈMẸ BLO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/1h/B17___02_Tite________GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_1h' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalno_vq_gungbe_1h" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <H3> 30mn </H3>


    <table>
        <thead>
            <tr>
                <td>#</td>
                <td>Texte </td>
                <td>Enregistrement original du Locuteur natif</td>
                <td>Données générées par le système</td>
                <td>Notes</td>
            </tr>
        </thead>
        <tbody>


            <tr>
                <td><span class="text"> 1 </span></td>
                <td><span class="text"> MALKU, ALISTALKU, DEMA, LUKU, AZÓNWÀTỌGBÉ ṢIE LẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00025.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00025_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 2 </span></td>
                <td><span class="text"> HIẸ HẸN ẸN WHÈ PẸVÌDE HÙ ANGELI LẸ; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___02_Hebreux_____GUWNVSN1DA_00007.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B19___02_Hebreux_____GUWNVSN1DA_00007_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 3 </span></td>
                <td><span class="text"> MIỌNHOMẸNA MI TO OKLUNỌ MẸ. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00021.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00021_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 4 </span></td>
                <td><span class="text"> HA DAGBEWÀ TOWE MA NA DO YIN NUMANANỌMAWÀ TỌN, ADAVO OJLO TỌN. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00013.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00013_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 5 </span></td>
                <td><span class="text"> ṢIGBA YEN MA YLO NADO WA NÚDE TO AYIHA TOWE MAYỌNẸN GODO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00012.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00012_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 6 </span></td>
                <td><span class="text"> ENẸWUTU NA HIẸ NI YÍ I, ENẸ WẸ, OVI OHÒ ṢIE MẸ TỌN: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 7 </span></td>
                <td><span class="text"> MẸHE YẸN SỌ GÒ DOHLAN; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00009.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00009_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 8 </span></td>
                <td><span class="text"> ENẸWUTU EYÍN HIẸ LẸN MI DO OHA MẸ, BO YÍ I DI YẸNLỌSU. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00016.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00016_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 9 </span></td>
                <td><span class="text"> YẸN PAULU KO YÍ ALỌ ṢIE TITI DO WLAN ẸN, YẸN NA SÚ I: .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B18___01_Philemon____GUWNVSN1DA_00018.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B18___01_Philemon____GUWNVSN1DA_00018_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 10 </span></td>
                <td><span class="text"> ṢIGBA MẸHE DÓ ONÚ POPO WE JIWHEYẸWHE. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00005.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B19___03_Hebreux_____GUWNVSN1DA_00005_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 11 </span></td>
                <td><span class="text"> NA YÈ HẸN MADOGÁN HE LÉDO EWLOSU GA. .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___05_Hebreux_____GUWNVSN1DA_00003.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B19___05_Hebreux_____GUWNVSN1DA_00003_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 12 </span></td>
                <td><span class="text"> HE MA NỌ DO DÓDÓ DAGBE LẸPO HIA; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00011.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B17___02_Tite________GUWNVSN1DA_00011_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 13 </span></td>
                <td><span class="text"> NA MẸDE DALI WẸ YÈ GBỌN DO DÓ OHÒ LẸ DOPODOPO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B19___03_Hebreux_____GUWNVSN1DA_00004.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B19___03_Hebreux_____GUWNVSN1DA_00004_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>


            <tr>
                <td><span class="text"> 14 </span></td>
                <td><span class="text"> BO SỌ NỌ DAGBÈDOGBÈMẸ BLO; .</span></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/original/B17___02_Tite________GUWNVSN1DA_00010.wav">
                    </audio></td>
                <td><audio controls="">
                        <source src="./fongbe_yoruba/no-vq/gungbe/30mn/B17___02_Tite________GUWNVSN1DA_00010_audio_priorgrad_0.wav">
                    </audio></td>
                <td><select id='state1' class='no_vq_gungbe_30mn' name='note[]'>
                        <option value="1">&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp; </option>
                        <option value="2">&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;</option>
                        <option value="3">&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;</option>
                        <option value="4">&nbsp;&nbsp;&nbsp;4&nbsp;&nbsp;&nbsp;</option>
                        <option value="5" selected>&nbsp;&nbsp;&nbsp;5&nbsp;&nbsp;&nbsp;</option>
                    </select> </td>
            </tr>

        </tbody>
        <tfoot>
            <tr>
                <td colspan="3"> </td>
                <td> Moyenne </td>
                <td class="total" id="totalno_vq_gungbe_30mn" data-avg=5><span class="text">5</span></td>
            </tr>
        </tfoot>

    </table>
    <!-- --------------#-------------  end tables group --------------#------------- -->

    <br /> <br />
    <br />
    <fieldset>
        <button type="submit" id='soumettre' value="Envoyer">Soumettre</button>
        <!-- <br><input type="button" value="submit" onclick="insert()">-->
    </fieldset>

</body>
<footer></footer>
<script type="text/javascript">
    $(document).ready(function() {
        showpopup = false;
        showonce = false;
        //console.log("#################INFO0");
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
            if (($("#sav_num").val()) == "" || ($("#sav_num").val().length) < 6
                //|| validatePhone($("#sav_num").val()) == false
            ) {
                alert("Veuillez fournir un numéro de télephone valide!");
                return;
            }

            $('.total').each(function(index, obj) {

                //you can use this to access the current item
                //$( this ).toggleClass( "example" );
                m_id = $(obj).attr('id'); //.prop('id')//id; //
                m_value = $(obj).attr('data-avg'); //children('span')[0].text(); //$('img', this)[0]
                //console.log("##send save key " + m_id + " == " + m_value);
                $.post("audiotext2.php", {
                        "c_name": m_id,
                        "c_value": m_value,
                        "c_number": $("#sav_num").val()
                    }, function() {
                        //console.log("Enregistré avec succes");
                        $("#sumessage").html("Enregistré avec succès").css({
                            "display": "block"
                        });
                        //$('body').scrollTo('#sumessage');
                        /*****ok ****** 
                         */
                        $("#sumessage").get(0).scrollIntoView({
                            behavior: 'smooth'
                        }); /***/
                        //document.getElementById("element-id").scrollIntoView();
                        $("#content_message").html("Enregistré avec succès");
                        $("#content_message").attr('data-message', "Succes");;
                        if (showpopup == false && showonce == false /*&& index > 0 && index % 10 == 0*/) {
                            $(".content").toggle();
                            showonce = true;
                            showpopup = true;
                        }
                        showpopup = true;
                        //showonce = true;

                    })
                    .done(function(data) {
                        //console.log(data);
                        //alert("second success");
                    })
                    .fail(function() {
                        alert("error");
                        showpopup = false;
                        showonce = true;
                    })
                    .always(function() {
                        //alert("finished");
                    });
            });
            //console.log("if Popup  == " + showpopup + " ==#" + $("#content_message").attr('data-message') + "#");
            /*if (showpopup == true || $("#content_message").attr('data-message') != "") {
                console.log("Popup");
                $(".content").toggle();
            }*/

        });

        // var classes = $('#divID').attr('class').split(' ');

        // for (var i = 0; i < classes.length; i++) {
        //     alert(classes[i]);
        // }
        // $("select[name*='note[]']").on('change', function() {
        //     console.log("Select on name change triggerred " + $(this).val());
        // });

        $('select').on('change', function() {
            //console.log("Select change triggerred " + $(this).val());
            classid = $(this).attr('class');
            totalid = "total" + classid;
            sommenote = 0;
            nbrelement = 0;
            $('.' + classid).each(function(index, obj) {
                sommenote = parseInt(sommenote) + parseInt($(obj).val()); //.value // alert( $(this).find(":selected").val() );$("#selectID option:selected").val(); //event.target.value 
                nbrelement = nbrelement + 1;
            });
            notemoyenne = (sommenote / nbrelement).toFixed(2);
            //console.log("### sommenote " + sommenote + " nbr " + nbrelement + " moyenne " + notemoyenne + " in " + $("#" + totalid).html());
            $("#" + totalid).attr('data-avg', notemoyenne);
            //$("#" + totalid).children('span')[0].html("" + notemoyenne);
            $("#" + totalid).html('<span class="text">' + notemoyenne + '</span>');
            //alert(this.value);
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

    function validatePhone(phone_number) {
        var a = phone_number; //document.getElementById(phone_number).value;
        var filter = /^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3,}[\s.-]\d{2,}$/; //^(\+0?1\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$
        if (filter.test(a)) {
            return true;
        } else {
            return false;
        }
    }

    function togglePopup() {
        $(".content").toggle();
        location.reload(true);
    }
</script>

</html>