<?php

namespace App\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Queue\SerializesModels;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Symfony\Component\Process\Process;

class AddFlight implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    protected $flightNbr;

    /**
     * Create a new job instance.
     *
     * @return void
     */
    public function __construct($flightNbr)
    {
        $this->flightNbr = $flightNbr;
    }

    /**
     * Execute the job.
     *
     * @return void
     */
    public function handle()
    {

        chdir(base_path('../'));

        $command = 'python3 spell_trainer.py -m flights -t ' . $this->flightNbr;

        $process = new Process($command);

        \Log::info('Started ' . $command);

        $process->setTimeout(null); // timeout in second
        
        $process->start(); // Start job async
        
        while ($process->isRunning()) {
            $process->checkTimeout();

            if ($message = $process->getOutput()) {
                \Log::info($message);
                sleep (10);
            }
        }

        if (!$process->isSuccessful()) {
            \Log::error($process->getErrorOutput());
        } else {
            \Log::info('Done training for ' . $this->flightNbr);
        }
    }
}
