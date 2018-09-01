<!doctype html>
<html lang="{{ app()->getLocale() }}">
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Flight Spell Checker</title>

        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css?family=Raleway:100,600" rel="stylesheet" type="text/css">

        
        <link href="{{mix('/css/app.css')}}" rel="stylesheet" type="text/css">

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

            [v-cloak] {display: none}
        </style>
    </head>
    <body>
        <div class="flex-center position-ref full-height">
            @if (Route::has('login'))
                <div class="top-right links">
                    @auth
                        <a href="{{ url('/home') }}">Home</a>
                    @else
                        <a href="{{ route('login') }}">Login</a>
                        <a href="{{ route('register') }}">Register</a>
                    @endauth
                </div>
            @endif

            <div class="content container" id="app" v-cloak>
                <div class="title justify-content-center">
                    Search Flights
                    <form class="form-inline" v-on:submit.prevent="addAnswer()">
                        <div class="form-group mb-4">
                            <input type="text" class="form-control" id="flight_nbr" placeholder="Flight No." v-model="question">
                            <button type="submit" class="btn btn-primary" v-bind:disabled="!question">Use this flight for training</button>
                        </div>
                    </form>
                </div>
                <div class="links" v-if="answers">
                    Found flights for:
                    <ul class="list-inline">
                        <li v-for="text in answers" class="list-inline-item">
                            <button class="btn btn-link" v-on:click.prevent="setQuestion(text)">@{{text}}</button>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    <script src="{{mix('/js/app.js')}}"></script>
</html>
