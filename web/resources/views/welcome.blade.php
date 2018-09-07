<!doctype html>
<html lang="{{ app()->getLocale() }}">
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Message Intelligence Project</title>

        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css?family=Raleway:100,600" rel="stylesheet" type="text/css">

        <!-- Styles -->
        <style>
            html, body {
                background-color: #fff;
                color: #636b6f;
                font-family: 'Raleway', sans-serif;
                font-weight: 100;
                height: 100vh;
                margin: 0;
            }

            .full-height {
                height: 100vh;
            }

            .flex-center {
                align-items: center;
                display: flex;
                justify-content: center;
            }

            .position-ref {
                position: relative;
            }

            .top-right {
                position: absolute;
                right: 10px;
                top: 18px;
            }

            .content {
                text-align: center;
            }

            .title {
                font-size: 84px;
            }

            .links > a {
                color: #636b6f;
                padding: 0 25px;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: .1rem;
                text-decoration: none;
                text-transform: uppercase;
            }

            .m-b-md {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="flex-center position-ref full-height">

            <div class="content">
                <div class="title m-b-md">
                    Unilode
                </div>

                <div class="links">
                    <a href="Javascript: botmanChatWidget.open()">Open Chatbot</a>
                    <!--
                        
                    |
                    <a href="{{url('flights')}}">Flight Checker</a>

                    -->
                </div>
            </div>
        </div>
        <script>
        var botmanWidget = {
            introMessage: '<img src="https://planb.unilode.com/assets/images/unilode_logo.svg"/><h4>Welcome to message intelligence</h4></i>',
            title: 'Unilode Chatbot',
            chatServer: '{{url('api/chatbot')}}',
            frameEndpoint: '{{url('chatbot')}}',
            aboutText: '',
            mainColor: '#0072c6'
        };
        </script>
        <script src='https://cdn.jsdelivr.net/npm/botman-web-widget@0/build/js/widget.js'></script>
    </body>
</html>
