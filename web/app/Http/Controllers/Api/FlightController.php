<?php

namespace App\Http\Controllers\Api;

use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use App\Jobs\AddFlight;

class FlightController extends Controller {

    public function __construct() {
        $this->client = new \GuzzleHttp\Client();
    }

    public function getFlights (Request $request) {
        $text = $request->input('text');
        $prob = $request->input('prob');

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

    public function addFlight (Request $request) {
        $flightNbr = $request->input('text');

        AddFlight::dispatch($flightNbr);
    }
}