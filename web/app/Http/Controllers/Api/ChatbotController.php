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
use BotMan\BotMan\Messages\Outgoing\Question;
use BotMan\BotMan\Messages\Incoming\Answer;
use BotMan\BotMan\Messages\Outgoing\Actions\Button;

class ChatbotController extends Controller {
    protected $models = ['flights', 'names', 'messages'];

    public function __construct() {
        DriverManager::loadDriver(\BotMan\Drivers\Web\WebDriver::class);

        $this->chatbot = BotManFactory::create([], new RedisCache('redis', 6379));

        $this->client = new \GuzzleHttp\Client();
    }

    public function chat() {
        $this->chatbot->hears('.*(Hi|Hello).*', 'App\Http\Controllers\Api\ChatbotController@handleGreeting');

        $this->chatbot->hears('.*(Bye|Good Bye|See you later|Talk to you later).*', 'App\Http\Controllers\Api\ChatbotController@handleBye');

        $this->chatbot->hears('(List|List command|List commands)', 'App\Http\Controllers\Api\ChatbotController@handleListCommands');

        $this->chatbot->hears('Check {text}', 'App\Http\Controllers\Api\ChatbotController@handleCheckText');
        
        $this->chatbot->fallback(function($bot) {
            $bot->reply('Sorry, I did not understand that.');
        });

        $this->chatbot->listen();
    }

    public function handleGreeting(Botman $bot) {
        $bot->reply('Hello there!');
        //$bot->reply('You can start checking flight number from our ML model. Start by type: Flight&nbsp;<i>Flight No.</i>');
    }

    public function handleBye(Botman $bot) {
        $bot->reply('Bye');
    }

    public function handleListCommands (Botman $bot) {
        $commands = '<ul>';

        foreach ($this->models as $m) {
            $commands .= '<li>Check ' . str_singular($m) .  '&nbsp;<i>{' . str_singular($m) . '}</i>' . '</li>';
        }

        $commands .= '</ul>';

        $bot->reply($commands);
    }

    public function handleCheckText (Botman $bot, $text) {
        $token = explode(' ', $text);

        if (count($token) > 1) {
            $model = str_plural($token[0]);
            $text = strtoupper($token[1]);
        } else {
            $model = null;
            $text = strtoupper($token[0]);
        }
        
        if (!$model || !in_array($model, $this->models)) {
            $bot->reply('I am not sure what to do with "' . $text . '".');
            $this->selectCommand($bot, $text);
        } else {
            $bot->reply($this->checkText($model, $text));
        }
    }

    public function checkText ($model, $text) {
        $reply = '';
        $answers = $this->checkApi($model, $text);

        if (in_array($text, $answers)) {
            $reply .= '<h3 style="color: green">Hooray!</h3> We\'ve found your ' . str_singular($model) . ': <br/>' . $text . '<br/>';
            $answers = array_diff($answers, [$text]);
            if (count($answers) > 0) {
                $reply .= 'We also found: <br/>'.implode('<br/>', $answers);
            }
        } else {
            $reply .= '<h3 style="color: red">Oops!</h3> Your search not found in ' . str_singular($model) . '. But we found: <br/>'.implode('<br/>', $answers);
        }
        
        return $reply;
    }

    public function selectCommand ($bot, $text) {
        $commands = array_map(function ($model) use ($text) {
            return Button::create('Check ' . str_singular($model) . ' ' . $text)->value($model);
        }, $this->models);

        $question = Question::create('Select what to check')->addButtons($commands);

        $that = $this;
        $bot->ask($question, function (Answer $answer) use ($that, $text) {
            if ($answer->isInteractiveMessageReply()) {
                $model = $answer->getValue();
                
                //$this->say('OK, ' . $answer->getText());
                $this->say($that->checkText($model, $text));
            } else {
                $this->say('Please select from above actions');
            }
        });
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