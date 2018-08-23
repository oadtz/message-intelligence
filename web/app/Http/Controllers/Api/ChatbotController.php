<?php

namespace App\Http\Controllers\Api;

use App\QuestionConversation;
use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use App\Jobs\AddFlight;
use BotMan\BotMan\BotMan;
use BotMan\BotMan\BotManFactory;
use BotMan\BotMan\Drivers\DriverManager;


class ChatbotController extends Controller {

    public function __construct() {
        DriverManager::loadDriver(\BotMan\Drivers\Web\WebDriver::class);

        $this->chatbot = BotManFactory::create([]);

        $this->client = new \GuzzleHttp\Client();
    }

    public function chat() {
        $this->chatbot->hears('.*(Hi|Hello).*', function (BotMan $bot) {
            $bot->reply('Hello there!');
            $bot->reply('You can start checking flight number from our ML model. Start by type: check&nbsp;<i>Flight No.</i>');
        });

        $this->chatbot->hears('.*(Bye|Good Bye|See you).*', function (BotMan $bot) {
            $bot->reply('Bye');
        });

        $this->chatbot->hears('Check ([0-9A-Za-z]+)', function ($bot, $flightNbr) {
            $answers = $this->checkFlights($flightNbr);
            
            $bot->reply('We found flight(s): <br/>'.implode('<br/>', $answers));
        });
        
        $this->chatbot->hears('Add ([0-9A-Za-z]+)', function ($bot, $flightNbr) {
            $this->addFlight($flightNbr);
            
            $bot->reply('We will add ' . $flightNbr . ' to our database. Thanks for your suggestion.');
        });

        $this->chatbot->hears('Delete', function ($bot) {
            $bot->startConversation(new QuestionConversation);
        });
        
        $this->chatbot->fallback(function($bot) {
            $bot->reply('Sorry, I did not understand that. Here is a list of commands I understand: <ul><li>Check&nbsp;<i>Flight No.</li><li>Add&nbsp;<i>Flight No.</li></ul>');
        });

        $this->chatbot->listen();
    }

    public function checkFlights($flightNbr) {

        $text = $flightNbr;
        $prob = 0.99;

        $answers = [];

        try {
            $data = compact('text', 'prob');

            $response = $this->client->request('GET', config('app.model_api_url') . '/flights', [ 'query' => $data ]);

            $answers = json_decode($response->getBody()->getContents());
        } catch (\Exception $e) {
            $answers = null;
        }

        return $answers;
    }

    public function addFlight ($flightNbr) {
        AddFlight::dispatch($flightNbr);
    }

}