<?php

namespace App\Http\Controllers\Api;

use App\QuestionConversation;
use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use App\Jobs\AddFlight;
use BotMan\BotMan\BotMan;
use BotMan\BotMan\BotManFactory;
use BotMan\BotMan\Drivers\DriverManager;
use BotMan\BotMan\Cache\RedisCache;
use BotMan\BotMan\Middleware\ApiAi;

class ChatbotController extends Controller {

    public function __construct() {
        DriverManager::loadDriver(\BotMan\Drivers\Web\WebDriver::class);

        $this->chatbot = BotManFactory::create([], new RedisCache('redis', 6379));

        $this->client = new \GuzzleHttp\Client();
    }

    public function chat() {
        $this->chatbot->hears('.*(Hi|Hello).*', function (BotMan $bot) {
            $bot->reply('Hello there!');
            $bot->reply('You can start checking flight number from our ML model. Start by type: Flight&nbsp;<i>Flight No.</i>');
        });

        $this->chatbot->hears('.*(Bye|Good Bye|See you).*', function (BotMan $bot) {
            $bot->reply('Bye');
        });

        $this->chatbot->hears('Do you know ([0-9A-Za-z]+)\?', function ($bot, $name) {
            $answers = $this->checkApi('names', $name);

            if (in_array($name, $answers)) {
                $bot->reply('<h3 style="color: green">Yes!</h3> I know ' . $name);
                $answers = array_diff($answers, [$name]);
                if (count($answers) > 0) {
                    $bot->reply('I also know '.implode(', ', $answers));
                }
            } else {
                $bot->reply('<h3 style="color: red">Sorry!</h3> I don\'t know any '. $name .'. But I know '.implode(', ', $answers));
            }
        });

        $this->chatbot->hears('Flight ([0-9A-Za-z]+)', function ($bot, $flightNbr) {
            $answers = $this->checkApi('flights', $flightNbr);

            if (in_array($flightNbr, $answers)) {
                $bot->reply('<h3 style="color: green">Hooray!</h3> We\'ve found your flight: <br/>' . $flightNbr);
                $answers = array_diff($answers, [$flightNbr]);
                if (count($answers) > 0) {
                    $bot->reply('We also found: <br/>'.implode('<br/>', $answers));
                }
            } else {
                $bot->reply('<h3 style="color: red">Oops!</h3> We have not your flight. But we found: <br/>'.implode('<br/>', $answers));
            }
        });
        
        $this->chatbot->hears('Add Flight ([0-9A-Za-z]+)', function ($bot, $flightNbr) {
            $this->addFlight($flightNbr);
            
            $bot->reply('We will add flight ' . $flightNbr . ' to our database. Thanks for your suggestion.');
        });

        $this->chatbot->hears('Message (.+)', function ($bot, $message) {
            $answers = $this->checkApi('messages', $message, 'POST', 1.0);

            $bot->reply(nl2br($answers[0]));
        });

        $this->chatbot->hears('Delete', function ($bot) {
            $bot->startConversation(new QuestionConversation);
        });
        
        $this->chatbot->fallback(function($bot) {
            $bot->reply('Sorry, I did not understand that. Here is a list of commands I understand: <ul><li>Flight&nbsp;<i>Flight No.</li><li>Add Flight&nbsp;<i>Flight No.</li></ul>');
        });

        $this->chatbot->listen();
    }

    public function checkApi($model, $text, $method = 'GET', $prob = 0.97) {

        $text = strtoupper($text);

        $answers = [];

        try {
            $data = compact('text', 'prob');

            if ($method == 'POST') {
                $response = $this->client->request('POST', config('app.model_api_url') . '/' . $model, [ 'form_params' => $data ]);
            } else {
                $response = $this->client->request('GET', config('app.model_api_url') . '/' . $model, [ 'query' => $data ]);
            }

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