<?php

namespace App;

use BotMan\BotMan\Messages\Conversations\Conversation;
use BotMan\BotMan\Messages\Outgoing\Question;
use BotMan\BotMan\Messages\Incoming\Answer;
use BotMan\BotMan\Messages\Outgoing\Actions\Button;

class QuestionConversation extends Conversation {

    public function run () {
        $question = Question::create('Confirm?')
            ->fallback('Oops! Something just wen wrong.')
            ->addButtons([
                Button::create('Yes')->value('yes'),
                Button::create('No')->value('no'),
            ]);

        $this->ask($question, function (Answer $answer) {
            // Detect if button was clicked:
            if ($answer->isInteractiveMessageReply()) {
                if ($answer->getValue() == 'yes') {
                    $this->say('You said Yes');
                    $this->say('But we\'ll do nothing.');
                } else {
                    $this->say('You said No');
                }
            } else {
                $this->say('You did not select a choice.');
            }
        });
    }

}